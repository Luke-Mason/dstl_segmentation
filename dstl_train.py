import argparse
import json
import math
import os
from sklearn.model_selection import KFold

import torch
import pandas as pd
import datetime
from torch.utils import tensorboard
from datasets import DSTLPreprocessor
from dataloaders import DSTLLoader
import models
from trainers import DSTLTrainer
from utils import Logger, losses, FilterConfig3D, BandGroup
import copy
from pathlib import Path
import numpy as np
import logging
from utils.metrics import (eval_metrics, recall, precision, f1_score,
                           pixel_accuracy, AverageMeter,
                           mean_average_precision, intersection_over_union)
from test_helper import dataset_gateway
import utils
from utils import metric_indx
from visualise_run_stats import obj_colors

torch.cuda.empty_cache()
import re
from PIL import Image
import numpy as np
import cv2
import tifffile as tiff
from visualise_run_stats import ids

def overlay_masks_on_image(src_image, target_mask, output_mask, output_path:
str, class_ids, experiment_id):
#%%
    h = None
    w = None

    # Dynamic range adjustment for the file
    # Thank you u1234x1234 for this dra code
    dra_image = cv2.cvtColor(src_image.transpose((1,2,0)), cv2.COLOR_RGB2BGR)
    dra_image = cv2.cvtColor(dra_image, cv2.COLOR_BGR2RGB)

    for c in range(dra_image.shape[2]):
        min_val, max_val = np.percentile(dra_image[:, :, c], [0.1, 99.9])
        dra_image[:, :, c] = 255 * (dra_image[:, :, c] - min_val) / (max_val - min_val)
        dra_image[:, :, c] = np.clip(dra_image[:, :, c], 0, 255)
    dra_image = (dra_image).astype(np.uint8)
    if h and w:
        dra_image = cv2.resize(dra_image, (w, h), interpolation=cv2.INTER_LANCZOS4)

    for class_pos, class_index in enumerate(class_ids):
        mask_out = output_mask[:, :, [class_pos]]
        class_name = metric_indx[str(class_ids[0])]
        mask_color = mask_out * hex_to_rgb(obj_colors[class_name])

        mask_target = target_mask[:, :, [class_pos]]
        target_color = mask_target * hex_to_rgb("#FFFFFF")

        combined_mask = np.bitwise_or(mask_out, mask_target)

        # Get inverse of mask out because we want to apply a addition
        # operator to the masked pixels so they plug together like two puzzle
        # pieces together to make 1 image.
        combined_mask_inv = np.invert(combined_mask.astype(np.bool_))

        # Get the mask pixels of the image with bitwise and operation
        rgb_mask_inv = np.bitwise_and(dra_image, combined_mask_inv * 255)
        # ac = np.zeros((rgb_mask_inv.shape[0], rgb_mask_inv.shape[1], 1),
        #                          dtype=rgb_mask_inv.dtype)
        # rgb_mask_inv = np.concatenate((rgb_mask_inv, ac), axis=2)
        # rgb_mask_inv[:, :, 3] = 1

        rgb_image = np.bitwise_and(dra_image, combined_mask * 255)
        rgb_image = np.bitwise_or(rgb_image, target_color)


        # ac = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 1),
        #                          dtype=rgb_image.dtype)
        # rgb_image = np.concatenate((rgb_image, ac), axis=2)
        # rgb_image[:, :, 3] = 1

        # ac = np.zeros((mask_color.shape[0], mask_color.shape[1], 1),
        #               dtype=mask_color.dtype)
        # mask_color = np.concatenate((mask_color, ac), axis=2)
        # mask_color[:, :, 3] = 0.5

        rgb_image = cv2.addWeighted(rgb_image, 0.5, mask_color, 0.5, 0)

        image = rgb_mask_inv + rgb_image
        image = Image.fromarray(image.astype(np.uint8))
        object_idx = class_ids[0] + 1
        type = "P" if class_index + 1 != 11 else "N"
        image.save(f"tmp/{ids[int(experiment_id) - 1]}-{object_idx}-"
                       f"{type}"
                       f"_{output_path}")
#%%

def hex_to_rgb(hex_color):
    # Remove the '#' character if present
    hex_color = hex_color.lstrip('#')

    # Convert the hexadecimal string to integers for R, G, and B
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)

def convert_mask_to_color(mask, class_ids):
    mask_color = np.zeros(mask.shape[1:] + (3,))
    mask_color = mask_color.astype(np.uint8)
    for class_pos, class_index in enumerate(class_ids):
        class_name = metric_indx[str(class_index)]
        colour = obj_colors[class_name]

        mask_color[mask[class_pos] == 1] = hex_to_rgb(colour)

    return mask_color


def get_preprocessor(config, root, _wkt_data, start_time,
                     train_indices, val_indices):
    training_band_groups = []
    preprocessing_ = config["all_loader"]['preprocessing']
    for group in preprocessing_['training_band_groups']:
        strategy = None
        filter3d = None
        if "filter_config" in group:
            filter3d = FilterConfig3D(group['filter_config']["kernel"],
                                      group['filter_config']["stride"])
        if "strategy" in group:
            strategy = group["strategy"]
        training_band_groups.append(
            BandGroup(group['bands'], filter3d, strategy)
        )

    # Preprocessing config
    preproccessing_config = copy.deepcopy(preprocessing_)
    del preproccessing_config['training_band_groups']

    return DSTLPreprocessor(
        root=root,
        data=_wkt_data,
        start_time=start_time,
        train_indices=train_indices,
        val_indices=val_indices,
        add_negative_class=config["all_loader"]["args"]["add_negative_class"],
        name=config['name'],
        training_band_groups=training_band_groups,
        save_dir=config["trainer"]["save_dir"],
        **preproccessing_config
    )


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def stratified_split(sorted_array, group_size):
    """
    Stratified split of a sorted array into groups of specified size.

    Parameters:
    sorted_array (list): The sorted array to be split.
    group_size (int): The size of each group.

    Returns:
    list: A list of lists representing the stratified split.
    """
    num_elements = len(sorted_array)
    num_groups = num_elements // group_size
    remainder = num_elements % group_size

    groups = []
    start_idx = 0

    for i in range(num_groups):
        group_end = start_idx + group_size
        groups.append(sorted_array[start_idx:group_end])
        start_idx = group_end

    # If there's a remainder, add it as the last group
    if remainder > 0:
        groups.append(sorted_array[start_idx:])

    return np.array(groups).transpose((1, 0))


def write_metric(logger, writer, do_validation, val_per_epochs, stats,
                 func, class_name):
    for key, m in stats.items():
        metric_name = key.capitalize()
        train_m1 = np.array(m['train'])
        val_m1 = np.array(m['val']) if 'val' in m else None
        for epoch in range(train_m1.shape[1]):
            metric_t = func(train_m1[:, epoch])
            metrics = dict({ 'train': np.mean(metric_t) })
            if do_validation and val_m1 is not None and (epoch + 1) % val_per_epochs == 0:
                val_epoch = epoch // val_per_epochs
                if val_epoch >= val_m1.shape[1]:
                    logger.error(f"val_epoch {val_epoch} is greater than val_m1.shape[1] {val_m1.shape[1]}")
                    continue
                metric_v = func(val_m1[:, val_epoch])
                val = dict({ 'val': np.mean(metric_v) })
                metrics.update(val)
            writer.add_scalars(f'{class_name}/{metric_name}', metrics, (epoch + 1))


def write_metric_2_param(logger, writer, do_validation, val_per_epochs, stats,
                         metric_1, metric_2, func, class_name, metric_name):
    m1, m2 = stats[metric_1], stats[metric_2]
    train_m1 = np.array(m1['train'])
    train_m2 = np.array(m2['train'])
    val_m1 = np.array(m1['val'])
    val_m2 = np.array(m2['val'])
    for epoch in range(train_m1.shape[1]):
        metric_t = func(train_m1[:, epoch], train_m2[:, epoch])
        train = dict({ 'train': np.mean(metric_t) })
        if do_validation and (epoch + 1) % val_per_epochs == 0:
            val_epoch = epoch // val_per_epochs
            metric_v = func(val_m1[:, val_epoch], val_m2[:, val_epoch])
            val = dict({ 'val': np.mean(metric_v) })
            train.update(val)
        writer.add_scalars(f'{class_name}/{metric_name}', train, (epoch + 1))

def write_metric_3_param(logger, writer, do_validation, val_per_epochs, stats,
                         metric_1, metric_2, metric_3, func, class_name,
                         metric_name):
    m1, m2, m3 = stats[metric_1], stats[metric_2], stats[metric_3]
    train_m1 = np.array(m1['train'])
    train_m2 = np.array(m2['train'])
    train_m3 = np.array(m3['train'])
    val_m1 = np.array(m1['val'])
    val_m2 = np.array(m2['val'])
    val_m3 = np.array(m3['val'])

    for epoch in range(train_m1.shape[1]):
        metric_t = func(train_m1[:, epoch], train_m2[:, epoch], train_m3[:, epoch])
        train = dict({ 'train': np.mean(metric_t) })
        if do_validation and ((epoch + 1)) % val_per_epochs == 0:
            val_epoch = epoch // val_per_epochs
            metric_v = func(val_m1[:, val_epoch], val_m2[:, val_epoch], val_m3[:, val_epoch])
            val = dict({ 'val': np.mean(metric_v) })
            train.update(val)
        writer.add_scalars(f'{class_name}/{metric_name}', train, (epoch + 1))

def write_stats_to_tensorboard(logger, writer, do_validation, val_per_epochs,
                               class_stats):
    # for stat in class_stats.keys():
    #     for metric in class_stats[stat].keys():
    #         for key in ['train', 'val']:
    #             logger.debug(f"{stat}-{metric}-{key}: {class_stats[stat][metric][key]}")

    # LOSS
    write_metric(logger, writer, do_validation, val_per_epochs, class_stats['all'], np.mean, 'All')

    for class_name_indx, stats in class_stats.items():
        class_name = metric_indx[str(class_name_indx)]

        # # mAP
        # write_metric(logger, writer, do_validation, val_per_epochs, stats, 'average_precision', np.mean, class_name, 'mAP')

        # PIXEL ACCURACY
        write_metric_2_param(logger, writer, do_validation, val_per_epochs,
                             stats, 'correct_pixels', 'total_pixels',
                             pixel_accuracy, class_name, 'Pixel_Accuracy')

        # PRECISION
        write_metric_2_param(logger, writer, do_validation, val_per_epochs, stats,
                             'intersection', 'predicted_positives',
                             precision, class_name, 'Precision')

        # RECALL
        write_metric_2_param(logger, writer, do_validation, val_per_epochs, stats,
                             'intersection', 'total_positives',
                             recall, class_name, 'Recall')

        # F1 SCORE
        write_metric_3_param(logger, writer, do_validation, val_per_epochs, stats,
                             'intersection', 'predicted_positives',
                                'total_positives', f1_score, class_name,
                                'F1_Score')

        # MEAN IoU
        write_metric_2_param(logger, writer, do_validation, val_per_epochs, stats,
                             'intersection', 'union',
                             intersection_over_union, class_name, 'Mean_IoU')

    # stat_key = 'correct_pixels'
    #
    # metric_stats = [{
    #         "metric": "Mean_IoU",
    #         "component_names": []
    #     },
    #     {
    #         "metric": "Pixel_Accuracy",
    #         "component_names": []
    #     },
    #     {
    #         "metric": "Precision",
    #         "component_names": []
    #     },
    #     {
    #         "metric": "Recall",
    #         "component_names": []
    #     },
    #     {
    #         "metric": "F1_Score",
    #         "component_names": []
    #     }
    # }]
    # for class_name_indx, stats in class_stats.items():
    #     for epoch in range(stats[stat_key][key].shape[1]):
    #         scalars = dict({})
    #         for key in ['train', 'val']:
    #             if key == 'val':
    #                 if do_validation and (epoch + 1) % val_per_epochs == 0:
    #                     val_epoch = epoch // val_per_epochs
    #                 else:
    #                     continue
    #             else:
    #                 val_epoch = epoch
    #
    #             acc = pixel_accuracy(stats[stat_key][key][:, val_epoch], stats[
    #             'total_labeled_pixels'][key][:, val_epoch])
    #             res = dict({ [key]: np.mean(acc)})
    #             scalars.update(res)
    #
    #
    #                     metric_v = func(val_m1[:, val_epoch], val_m2[:, val_epoch])
    #                     val = dict({'val': np.mean(metric_v)})
    #                     scalars.update(val)
    #
    #
    #
    # writer.add_scalars(f'{metric_name}/{class_name}', scalars, (epoch + 1))


def main(config, model_pth, run_model: bool):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # This was made an environment variable and not in config because when
    # testing and running multiple config files on one machine is frustration
    # to update the config file each time you want to run it on a different
    # machine, like the gpu cluster that has a different file system or the
    # data exists elsewhere from the development environment.
    dstl_data_path = os.environ.get('DSTL_DATA_PATH')
    if dstl_data_path is None:
        raise EnvironmentError(
            'DSTL_DATA_PATH environment variable is not set, '
            'it must be a path to your DSTL data directory.')

    # Load the CSV into a DataFrame
    df = pd.read_csv(
        os.path.join(dstl_data_path, 'train_wkt_v4.csv/train_wkt_v4.csv'))

    # Get the data metadata list.
    _wkt_data = {}
    for index, row in df.iterrows():
        im_id = row['ImageId']
        class_type = row['ClassType']
        poly = row['MultipolygonWKT']

        # Add the polygon to the dictionary
        _wkt_data.setdefault(im_id, {})[int(class_type)] = poly

    image_ids = list(_wkt_data.keys())
    training_classes_ = config['all_loader']['preprocessing']['training_classes']

    # Stratified K-Fold
    mask_stats = json.loads(Path('dataloaders/labels/dstl-stats.json')
                            .read_text())

    im_area = [(idx, np.mean([mask_stats[im_id][str(cls)]['area'] for cls
                                in training_classes_]))
               for idx, im_id in enumerate(image_ids)]

    sorted_by_area = sorted(im_area, key=lambda x: x[1], reverse=True)
    sorted_by_area = [t[0] for t in sorted_by_area]
    logger.debug(f"Sorted Area {sorted_by_area}")

    sorted_by_area = dataset_gateway(sorted_by_area)
    _wkt_data = [(idx, key, value) for idx, (key, value) in enumerate(list(_wkt_data.items()))]
    if run_model:
        highest_class_in_image_idx = sorted_by_area[0]
        sorted_by_area = [highest_class_in_image_idx]
        _wkt_data = [(idx, key, value) for (idx, key, value) in _wkt_data if idx == highest_class_in_image_idx]
    else:
        _wkt_data = [(idx, key, value) for (idx, key, value) in _wkt_data if idx in sorted_by_area]
        split_count = config["trainer"]["k_split"] if len(im_area) > config["trainer"]["k_split"] else len(im_area)
        arr = stratified_split(sorted_by_area, split_count)
        stratisfied_indices = arr.flatten()

    # LOSS
    loss = getattr(losses, config['loss'])(weights=config['channel_weights'], threshold=config['threshold'])
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

    if config["trainer"]["val"] and not run_model:
        # Split the data into K folds
        shuffle_ = config["trainer"]["k_shuffle"]
        random_state_ = config["trainer"]["k_random_state"] if shuffle_ else None
        kfold = KFold(n_splits=split_count, shuffle=shuffle_,
                      random_state=random_state_)

        area_by_id = dict(im_area)

        preprocessing_ = config['all_loader']['preprocessing']
        training_classes_str = '_'.join(
            str(i) for i in preprocessing_['training_classes'])
        training_band_groups = [f"({'_'.join(map(str, band_group['bands']))})"
                                for band_group in
                                preprocessing_['training_band_groups']]
        run_name = (f"batch_size_{config['all_loader']['args']['batch_size']}"
                    f"_lr_{config['optimizer']['args']['lr']}"
                    f"_k_stop_{str(config['trainer']['k_stop'])}"
                    f"_epochs_{config['trainer']['epochs']}"
                    f"_loss_{config['loss']}"
                    f"_scheduler_{config['lr_scheduler']['type']}"
                    f"_patch_size_{preprocessing_['patch_size']}"
                    f"_overlap_pixels_{preprocessing_['overlap_pixels']}"
                    f"_training_classes_({training_classes_str})"
                    f"_training_band_groups_[{'-'.join(training_band_groups)}]")
        writer_dir = os.path.join(config['trainer']['log_dir'], config['name'], run_name, start_time)
        writer = tensorboard.SummaryWriter(writer_dir)

        # Initialise the stats
        fold_stats = dict()

        # Iterate over the K folds
        bonus = 0
        for fold_indx, (train_indxs_of_indxs, val_indxs_of_indxs) in enumerate(kfold.split(stratisfied_indices)):
            logger.info(f'Starting Fold {fold_indx + 1}:')
            train_indxs = stratisfied_indices[train_indxs_of_indxs]
            val_indxs = stratisfied_indices[val_indxs_of_indxs]

            # Logging
            logger.info(f"Train: {' '.join([str(k) for k in train_indxs])}")
            logger.info(f"Valid: {' '.join([str(k) for k in val_indxs])}")
            logger.info(
                f'Train area mean: {np.mean([area_by_id[im_id] for im_id in train_indxs]):.6f}')
            logger.info(
                f'Valid area mean: {np.mean([area_by_id[im_id] for im_id in val_indxs]):.6f}')
            train_area_by_class, valid_area_by_class = [
                {cls: np.mean(
                    [mask_stats[image_ids[im_id]][str(cls)]['area'] for im_id
                     in im_ids])
                    for cls in training_classes_}
                for im_ids in [train_indxs, val_indxs]]

            logger.info(f"Train area by class: "
                        f"{' '.join(f'{cls}: {train_area_by_class[cls]:.6f}' for cls in training_classes_)}")
            logger.info(f"Valid area by class: "
                        f"{' '.join(f'cls-{cls}: {valid_area_by_class[cls]:.6f}' for cls in training_classes_)}")

            preprocessor = get_preprocessor(config, dstl_data_path,
                                            _wkt_data, start_time,
                                            train_indices=train_indxs,
                                            val_indices=val_indxs)

            train_patch_files, val_files = preprocessor.get_files()

            logger.info("Creating file weights..")
            train_patch_weights, val_patch_weights = preprocessor.get_file_weights()

            # Check if train or val weights contains Inf or NaN
            if np.any(np.isnan(train_patch_weights)) or np.any(np.isnan(val_patch_weights)):
                logger.error("Train or val weights contains Inf")
                bonus += 1
                continue

            # Create train and valiation data loaders that only load the data
            # into batch by seleecting indexes from the list of indices we
            # give each loader.
            all_loader_config = {
                "batch_size": config["all_loader"]["args"]["batch_size"],
                "num_workers": config["all_loader"]["args"]["num_workers"],
                "return_id": config["all_loader"]["args"]["return_id"],
            }
            logger.info("Creating loaders..")

            train_loader = DSTLLoader(
                **all_loader_config,
                files=train_patch_files,
                weights=train_patch_weights,
                **config["train_loader"]["args"]
            )
            val_loader = DSTLLoader(
                **all_loader_config,
                files=val_files,
                weights=val_patch_weights,
                **config["val_loader"]["args"]
            )

            add_negative_class = config["all_loader"]["args"]["add_negative_class"]
            negative_class_bonus = 1 if add_negative_class else 0
            # MODEL
            num_classes = len(training_classes_) + negative_class_bonus
            model = get_instance(models, 'arch', config, num_classes)

            logger.info("Creating trainer..")

            # TRAINING
            trainer = DSTLTrainer(
                start_time=start_time,
                k_fold=fold_indx,
                model=model,
                loss=loss,
                resume=model_pth,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                writer=writer,
                train_logger=logger,
                root=dstl_data_path,
                training_classes=training_classes_,
                do_validation=config['trainer']['val'],
                num_classes=num_classes,
                add_negative_class=add_negative_class,
            )

            try:
                logger.info("Train..")
                epochs_stats = trainer.train()
            except Exception as e:
                logger.error(f"Error in fold {fold_indx + 1}: {e}")
                bonus += 1
                continue

            # logger.debug(f"Fold stats BLALBLBLALSDLALSDLASLDLASD")
            # im lazy and dont want to refactor the code
            # class, metric, mode, epochs
            for class_name, class_stats in epochs_stats.items():
                if class_name not in fold_stats:
                    fold_stats[class_name] = dict()
                for metric_name, metric_stats in class_stats.items():
                    if metric_name not in fold_stats[class_name]:
                        fold_stats[class_name][metric_name] = dict()
                    for type, all_epoch_stats in metric_stats.items():
                        if type not in fold_stats[class_name][metric_name]:
                            fold_stats[class_name][metric_name][type] = []
                        # logger.debug(f"SHAPE {np.array(all_epoch_stats).shape}, {all_epoch_stats}")
                        fold_stats[class_name][metric_name][type].append(all_epoch_stats)

            # logger.debug(f"Fold stats: {fold_stats}")
            #
            # logger.info(f'Finished Fold {fold_indx + 1}:')
            if (config["trainer"]["k_stop"] is not None and config["trainer"][
                "k_stop"] > 0 and config["trainer"]["k_stop"] + bonus ==
                    fold_indx + 1):
                break

        # Write the stats to tensorboard
        # logger.info(f"val per epoch!!!!!! {config['trainer']['val_per_epochs']}")
        if not fold_stats:
            logger.error("Class stats is empty")
            return
        write_stats_to_tensorboard(logger, writer, config['trainer']['val'],
                                   config['trainer']['val_per_epochs'],
                                   fold_stats)

    else:
        preprocessor = get_preprocessor(config, dstl_data_path,
                                        _wkt_data, start_time,
                                        train_indices=[],
                                        val_indices=sorted_by_area)

        __, val_files = preprocessor.get_files()
        print(f"FILE LEN: {len(val_files)}")

        config['all_loader']['args']['batch_size'] = len(val_files)

        # Create train and valiation data loaders that only load the data
        # into batch by seleecting indexes from the list of indices we
        # give each loader.
        all_loader_config = {
            "batch_size": config["all_loader"]["args"]["batch_size"],
            "num_workers": config["all_loader"]["args"]["num_workers"],
            "return_id": config["all_loader"]["args"]["return_id"],
        }
        logger.info("Creating loaders..")

        data_loader = DSTLLoader(
            **all_loader_config,
            files=val_files,
            run_model=True,
            **config["val_loader"]["args"]
        )

        add_negative_class = config["all_loader"]["args"]["add_negative_class"]
        negative_class_bonus = 1 if add_negative_class else 0
        # MODEL
        num_classes = len(training_classes_) + negative_class_bonus
        model = get_instance(models, 'arch', config, num_classes)

        logger.info("Creating trainer..")

        # TRAINING
        trainer = DSTLTrainer(
            start_time=start_time,
            model=model,
            loss=loss,
            resume=model_pth,
            config=config,
            val_loader=data_loader,
            train_logger=logger,
            root=dstl_data_path,
            training_classes=training_classes_,
            do_validation=config['trainer']['val'],
            num_classes=num_classes,
            add_negative_class=add_negative_class,
        )

        outputs, targets = trainer.train()

        idx, image_id, __ = _wkt_data[0]
        tc = training_classes_ + [10] if add_negative_class else (
            training_classes_)

        # file name stuff
        pattern = r'saved/models/ex(.+)-.+/'

        # Use re.search to find the pattern in the input string
        match = re.search(pattern, model_pth)

        # Check if a match was found
        if match:
            # Extract the matched number from the regex group
            experiment_id = match.group(1)
        else:
            raise ValueError(f'No match found for {pattern} in {model_pth}')

        patches = {
            0: 54,
            1: 33,
            2: 154,
            3: 175,
            4: 234,
            5: 122,
            6: 121,
            7: 133,
            8: 70,
            9: 109,
        }

        idx_allowed = patches[tc[0]]
        for idx, (image_patch, output_mask, target_mask) in (
                enumerate(zip(val_files, outputs, targets))):
            if idx_allowed != idx:
                continue

            output_mask = output_mask.cpu().numpy().transpose(1, 2, 0)
            target_mask = target_mask.cpu().numpy().transpose(1, 2, 0)
            overlay_masks_on_image(image_patch[0], target_mask, output_mask,
                                   str(f".png"), tc, experiment_id)

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-r', '--run', default=False, type=bool,
                        help='Whether to do validation')
    parser.add_argument('-l', '--cl', default=None, type=int,
                        help='A specific class to train on, overriting config')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.cl is not None:
        config["all_loader"]["preprocessing"]["training_classes"] = [args.cl]

    if "training_classes" not in config["all_loader"]["preprocessing"]:
        raise ValueError("Training classes is None")

    print(f"Running experiment for class {args.cl}...")

    main(config, args.model, args.run)

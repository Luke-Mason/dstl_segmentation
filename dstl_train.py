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

torch.cuda.empty_cache()


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


def main(config, resume):
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

    _wkt_data = list(_wkt_data.items())
    _wkt_data = dataset_gateway(_wkt_data)

    training_classes_ = config['all_loader']['preprocessing'][
        'training_classes']

    # Stratified K-Fold
    mask_stats = json.loads(Path(
        'dataloaders/labels/dstl-stats.json').read_text())
    image_ids = df['ImageId'].unique()
    im_area = [(idx, np.mean([mask_stats[im_id][str(cls)]['area'] for cls
                                in training_classes_]))
               for idx, im_id in enumerate(image_ids)]

    im_area = dataset_gateway(im_area)

    sorted_by_area = sorted(im_area, key=lambda x: str(x[1]), reverse=True)
    sorted_by_area = [t[0] for t in sorted_by_area]
    logger.debug(f"Sorted Area {sorted_by_area}")
    split_count = config["trainer"]["k_split"] if len(im_area) > config["trainer"]["k_split"] else len(im_area)
    arr = stratified_split(sorted_by_area, split_count)
    stratisfied_indices = arr.flatten()

    # LOSS
    loss = getattr(losses, config['loss'])(threshold=config['threshold'])
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

    if config["trainer"]["val"]:
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

            train_patch_files, val_patch_files = preprocessor.get_files()

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
                files=val_patch_files,
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
                resume=resume,
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
        # DATA LOADERS
        train_loader = get_loader_instance('train_loader', _wkt_data, config, start_time)
        val_loader = get_loader_instance('val_loader', _wkt_data, config, start_time)

        # MODELMODEL
        model = get_instance(models, 'arch', config, len(training_classes_) + 1)

        # TRAINING
        trainer = DSTLTrainer(
            start_time=start_time,
            model=model,
            loss=loss,
            resume=resume,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_logger=logger,
            root=dstl_data_path,
            training_classes=training_classes_,
            do_validation=config['trainer']['val'],
            num_classes=len(training_classes_) + 1,
            add_negative_class=config["all_loader"]["args"]["add_negative_class"],
            k_fold=0,
        )

        trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
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

    main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)

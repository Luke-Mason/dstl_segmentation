import datetime
import json
import os
import time
import datetime
import numpy as np
import torch
from base import BaseTrainer
from torchvision import transforms
from torchvision.utils import make_grid, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import (eval_metrics, recall, precision, f1_score,
                           pixel_accuracy, AverageMeter,
                           mean_average_precision, intersection_over_union)
from utils import metric_indx
import logging
from utils import transforms as local_transforms


class DSTLTrainer(BaseTrainer):
    def __init__(self, start_time, model, loss, resume, config, train_loader,
                 val_loader,
                 writer,
                 num_classes,
                 add_negative_class,
                 k_fold, do_validation, training_classes, train_logger=None,
                 root='.'):

        super(DSTLTrainer, self).__init__(start_time, model, loss, resume,
                                          config, train_loader, val_loader,
                                          writer, k_fold, do_validation,
                                          training_classes,
                                          train_logger, root)
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(
            self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(
            self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = num_classes
        self.add_negative_class = add_negative_class
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            transforms.Resize((400, 400), antialias=True),
            transforms.Grayscale(num_output_channels=1),
        ])

        self.vis_transform = transforms.Compose([
            # local_transforms.DeNormalize(self.train_loader.MEAN,
            #                              self.train_loader.STD),
            # transforms.ToPILImage(),
            transforms.Resize((400, 400), antialias=True),
        ])


        self.min_clip_percentile = 5
        self.max_clip_percentile = 95

        torch.backends.cudnn.benchmark = True
        self.threshold = config["threshold"]

    def dra(self, array: np.ndarray) -> np.ndarray:
        output = np.zeros(array.shape, np.uint8)
        mask = array[0, :, :] != 0
        for i in range(1, array.shape[0] - 1):
            mask &= array[i, :, :] != 0

        for i in range(array.shape[0]):
            masked_array = array[i][mask]
            min_pixel = np.percentile(masked_array, self.min_clip_percentile)
            max_pixel = np.percentile(masked_array, self.max_clip_percentile)
            array[i] = array[i].clip(min_pixel, max_pixel)

            array[i] -= array[i].min()
            output[i] = array[i] / (array[i].max() / 255)

        return output

    def dra3(self, array: torch.Tensor) -> torch.Tensor:
        # Convert array to float tensor
        array = array.float()

        output = torch.zeros(array.shape, dtype=torch.uint8)
        mask = array[0, :, :] != 0
        for i in range(1, array.shape[0] - 1):
            mask &= array[i, :, :] != 0

        for i in range(array.shape[0]):
            masked_array = array[i][mask]
            min_pixel = torch.quantile(masked_array, self.min_clip_percentile
                                       / 100)
            max_pixel = torch.quantile(masked_array, self.max_clip_percentile
                                       / 100)
            array[i] = torch.clamp(array[i], min_pixel, max_pixel)

            array[i] -= array[i].min()
            output[i] = (array[i] / (array[i].max() / 255)).to(torch.uint8)

        return output

    def dra2(self, array: np.ndarray):
        # Calculate the values at the specified percentiles
        min_clip_value = np.percentile(array, self.min_clip_percentile)
        max_clip_value = np.percentile(array, self.max_clip_percentile)

        # Clip the values outside the specified range
        adjusted_array = np.clip(array, min_clip_value, max_clip_value)

        # Normalize the values to [0, 1] range
        adjusted_array = (adjusted_array - min_clip_value) / (
                max_clip_value - min_clip_value)

        return adjusted_array

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()

        batch_loss_history = np.array([])
        batch_learning_rates = [np.array([]) for _ in enumerate(self.optimizer.param_groups)]

        epoch_metrics = dict()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            # LOSS & OPTIMIZE
            output = self.model(data)
            target = target.to(self.device)

            if self.config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[1:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[1:] == target.size()[1:]
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch < self.config["lr_scheduler"]["args"]["stop_epoch"]:
                self.lr_scheduler.step()

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                batch_loss_history = np.append(batch_loss_history, loss.item())

            for i, opt_group in enumerate(self.optimizer.param_groups):
                # self.logger.debug(f"\nLearning {i}: {opt_group['lr']}")
                batch_learning_rates[i] = np.append(batch_learning_rates[i], opt_group['lr'])
                # self.logger.debug(f"\n Size Learning {i}: {len(batch_learning_rates[i])}")

            # Caluclate metrics for all classes
            batch_metrics = eval_metrics(output, target, self.threshold)
            if 'all' not in epoch_metrics:
                epoch_metrics['all'] = batch_metrics
            else:
                for metric, total in batch_metrics.items():
                    epoch_metrics['all'][metric] += total

            # Caluclate metrics for each class
            for class_idx in range(self.num_classes):
                class_batch_metrics = eval_metrics(
                    output[:, class_idx, :, :][:, np.newaxis, :, :],
                    target[:, class_idx, :, :][:, np.newaxis, :, :],
                                                    self.threshold)
                # Mapping class_indx to class_name_indx i.e 0,1,2 into 2,5,9
                extra_negative_class = 1 if self.add_negative_class == True else 0
                class_name_idx = self.training_classes[class_idx] if class_idx < len(self.training_classes) else 9 + extra_negative_class
                if str(class_name_idx) not in epoch_metrics:
                    epoch_metrics[str(class_name_idx)] = class_batch_metrics
                else:
                    for metric, total in class_batch_metrics.items():
                        epoch_metrics[str(class_name_idx)][metric] += total

            # TODO


            # PRINT INFO
            seg_metrics = self._get_metrics(batch_metrics)
            tbar.set_description(f'TRAIN EPOCH {epoch} | Batch: {batch_idx + 1} | ')
            message = (f'\nTRAIN EPOCH {epoch} | Batch: {batch_idx + 1} | Loss: {loss.item():.3f} | ')
            for metric, total in seg_metrics.items():
                message += f'{metric}: {total:.3f} | '
            self.logger.info(message)

        self.logger.info(f"Finished training epoch {epoch}")

        # Add loss
        epoch_metrics['all']['loss'] = batch_loss_history
        # self.logger.debug(f"Learning Group Shape: {np.array(self.optimizer.param_groups).shape}")
        # self.logger.debug(f"Learning Group Shape 0:"
        #                   f" {np.array(self.optimizer.param_groups[0]).shape}")
        # self.logger.debug(f"Learning Group 0 keys: {self.optimizer.param_groups[0]['lr']}")

        for i, opt_group in enumerate(self.optimizer.param_groups):
            epoch_metrics['all'][f'lr_{i}'] = batch_learning_rates[i]

        return epoch_metrics

    def convert_to_title_case(self, input_string):
        words = input_string.split('_')
        capitalized_words = [word.capitalize() for word in words]
        return ' '.join(capitalized_words)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning(
                'Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        loss_history = np.array([])
        total_metric_totals = dict()

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tbar):
                # LOSS
                output = self.model(data)
                target = target.to(self.device)

                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()

                if batch_idx % self.log_step == 0:
                    loss_history = np.append(loss_history, loss.item())

                # METRICS
                metrics_totals = eval_metrics(output, target, self.threshold)
                if 'all' not in total_metric_totals:
                    total_metric_totals['all'] = metrics_totals
                else:
                    for k, v in metrics_totals.items():
                        total_metric_totals['all'][k] += v

                for class_idx in range(self.num_classes):
                    class_metrics_totals = eval_metrics(
                        output[:, class_idx, :, :][:, np.newaxis, :, :],
                        target[:, class_idx, :, :][:, np.newaxis, :, :],
                                                    self.threshold)
                    # Convert class_indx into class_name_indx
                    extra_negative_class = 1 if self.add_negative_class == True else 0
                    class_name_idx = self.training_classes[class_idx] if class_idx < len(self.training_classes) else 9 + extra_negative_class
                    if str(class_name_idx) not in total_metric_totals:
                        # Convert class_indx into class_name_indx

                        total_metric_totals[str(class_name_idx)] = class_metrics_totals
                    else:
                        for k, v in class_metrics_totals.items():
                            total_metric_totals[str(class_name_idx)][k] += v

                # PRINT INFO
                seg_metrics = self._get_metrics(metrics_totals)
                description = f'EVAL EPOCH {epoch} | Batch: {batch_idx + 1} | '
                for k, v in seg_metrics.items():
                    description += f'{self.convert_to_title_case(k)}: {v:.3f} | '
                tbar.set_description(description)

                # WRTING & VISUALIZING THE MASKS
                # LIST OF IMAGE TO VIZ (15 images)
                if batch_idx % 20 == 0 and self.k_fold == 0:
                    for k in range(1):
                        dta, tgt, out = data[k], target[k], output[k]

                        dta = dta * 2047
                        dta = torch.tensor(dta).to(self.device)
                        _dta = self.vis_transform(dta.to(torch.uint8))
                        dta = torch.unsqueeze(_dta, dim=0)
                        for i in range(self.num_classes):
                            tgi = tgt[i, :, :][np.newaxis, :, :]
                            tgi = (tgi > self.threshold).float().to(torch.int) * 255
                            tgi = self.restore_transform(tgi.to(torch.uint8))

                            outi = out[i, :, :][np.newaxis, :, :]
                            outi = (outi > self.threshold).float().to(torch.int) * 255
                            outi = self.restore_transform(outi.to(torch.uint8))

                            # %%
                            # plt.figure(figsize=(20, 7))
                            # plt.subplot(2, 4, 1)
                            # plt.title('Input (Image)')
                            # plt.imshow(torch.from_numpy(self.dra(
                            #     _dta.numpy())).permute(1, 2, 0))
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 2)
                            # plt.title('Input (Image)')
                            # plt.imshow(torch.from_numpy(self.dra2(
                            #     _dta.numpy())).permute(1, 2, 0))
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 3)
                            # plt.title('Input (Image)')
                            # plt.imshow(self.dra3(_dta).permute(1, 2, 0))
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 4)
                            # plt.title('Input (Image)')
                            # plt.imshow(_dta.permute(1, 2, 0)[:, :, [1,2,0]])
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 5)
                            # plt.title('Input (Image)')
                            # plt.imshow(_dta.permute(1, 2, 0)[:, :, [2,0,1]])
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 6)
                            # plt.title('Input (Image)')
                            # plt.imshow(_dta.permute(1, 2, 0)[:, :, [2,1,0]])
                            # plt.axis('off')
                            #
                            # plt.subplot(2, 4, 7)
                            # plt.title('Output (Predicted)')
                            # plt.imshow(outi.permute(1,2,0), cmap='gray')
                            # plt.axis('off')
                            #
                            # # Plotting the target tensor (ground truth)
                            # plt.subplot(2, 4, 8)
                            # plt.title('Target (Ground Truth)')
                            # plt.imshow(tgi.permute(1,2,0), cmap='gray')
                            # plt.axis('off')
                            #
                            # plt.tight_layout()
                            # plt.show()
                            # %%

                            tgi = torch.unsqueeze(tgi, dim=0)
                            tgi = tgi.expand(-1, 3, -1, -1)

                            outi = torch.unsqueeze(outi, dim=0)
                            outi = outi.expand(-1, 3, -1, -1)

                            imgs = torch.cat([dta, tgi, outi], dim=0)
                            grid_img = make_grid(imgs, nrow=3)

                            # Get class name from the class index
                            #  TODO
                            extra_negative_class = 1 if self.add_negative_class == True else 0
                            class_name_idx = self.training_classes[i] if i < len(self.training_classes) else 9 + extra_negative_class
                            class_name = metric_indx[str(class_name_idx)]
                            # row shows one class (num_classes_to_predict)
                            self.writer.add_image(
                                f'inputs_targets_predictions/{class_name}',
                                                  grid_img, epoch)

        # Add loss
        total_metric_totals['all']['loss'] = loss_history

        return total_metric_totals

    def _get_metrics(self, seg_class_totals):
        seg_totals = seg_class_totals['all'] if 'all' in seg_class_totals else seg_class_totals
        pixAcc = pixel_accuracy(seg_totals['correct_pixels'], seg_totals['total_pixels'])
        p = precision(seg_totals['intersection'], seg_totals['predicted_positives'])
        r = recall(seg_totals['intersection'], seg_totals['total_positives'])
        f1 = f1_score(seg_totals['intersection'], seg_totals['predicted_positives'], seg_totals['total_positives'])
        # mAP = mean_average_precision(seg_totals['average_precision'])
        mIoU = intersection_over_union(seg_totals['intersection'], seg_totals['union'])

        return {
            "Mean_IoU": np.round(mIoU, 3),
            # "mAP": np.round(mAP, 3),
            "F1": np.round(f1, 3),
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Precision": np.round(p, 3),
            "Recall": np.round(r, 3),
        }

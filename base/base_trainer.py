import os
import logging
import json
import math
import torch
import datetime
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
import numpy as np
from test_helper import epoch_gateway

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:

    def __init__(self,
                 start_time, model, loss, resume, config,
                 train_loader,
                 val_loader,
                 writer, k_fold, do_validation, training_classes,
                 train_logger=None, root='.'):
        self.root = root
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self._setup_logging()
        self.do_validation = do_validation
        self.start_epoch = 1
        self.improved = False
        self.k_fold = k_fold
        self.writer = writer
        self.training_classes = training_classes

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        print("AVAILABLE: ", availble_gpus, "DEVICE: ", self.device)
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = epoch_gateway(cfg_trainer['epochs'])
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                                    'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 
                                    'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        print(config['lr_scheduler']['args'])
        print("ITERATIONS PER EPOCH: ", len(train_loader))
        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(optimizer=self.optimizer,
                                     num_epochs=self.epochs,
                                     _iters_per_epoch=len(train_loader),
                                     **config['lr_scheduler']['args']
                                     )

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSORBOARD


        fold_name = (f"{'K_' + str(k_fold) if k_fold is not None else 'run'}")

        path = os.path.join(self.config['name'], start_time, fold_name)

        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'],
                                           "checkpoints", path)
        helpers.dir_exists(self.checkpoint_dir)

        self.run_dir = os.path.join(cfg_trainer['save_dir'], "run_info", path)

        helpers.dir_exists(self.run_dir)

        config_save_path = os.path.join(self.run_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        if resume: self._resume_checkpoint(resume)

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Logs to file

        # Check if log directory exist, if not create it
        log_dir = os.path.join(self.root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H.log')
        handler = logging.FileHandler(os.path.join(log_dir, log_file_name),
                                      mode='a')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        print('Cuda is available?: ', torch.cuda.is_available())
        print('Count of using GPUs:', torch.cuda.device_count())
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    
    def train(self):
        stats = dict()

        for epoch in range(self.start_epoch, self.epochs+1):
            # RUN TRAIN (AND VAL)
            epoch_stats = self._train_epoch(epoch)
            for class_name, metric_totals in epoch_stats.items():
                if class_name not in stats:
                    stats[class_name] = dict()
                for metric_name, total in metric_totals.items():
                    if metric_name not in stats[class_name]:
                        stats[class_name][metric_name] = dict({
                            # List because I append arrays into it
                            'train': [],
                            'val': []
                        })
                    stats[class_name][metric_name]['train'].append(total)
            metrics = {}
            if (self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0):
                epoch_stats = self._valid_epoch(epoch)
                for class_name, metric_totals in epoch_stats.items():
                    if class_name not in stats:
                        stats[class_name] = dict()
                    for metric_name, total in metric_totals.items():
                        if metric_name not in stats[class_name]:
                            stats[class_name][metric_name] = dict({
                                'train': [],
                                'val': []
                            })
                        stats[class_name][metric_name]['val'].append(total)

                # LOGGING INFO
                self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
                for k, class_stats in epoch_stats.items():
                    self.logger.info(f'\n    Class {k}: ')
                    metrics = self._get_metrics(class_stats)
                    for q, p in metrics.items():
                        self.logger.info(f'         {str(q):15s}: {p}')
            all_metrics = self._get_metrics(epoch_stats)
            if self.train_logger is not None:
                log = {'epoch' : epoch, **metrics}
                self.train_logger.info(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            scheduler_args_ = self.config['lr_scheduler']["args"]
            last_epoch_passed =  epoch > scheduler_args_['stop_epoch'] if "stop_epoch" in scheduler_args_ else True
            if last_epoch_passed and self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    # TODO Seg metrics, check result here
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break
                    
                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stopped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

        self.logger.debug(f'Epoch {epoch} stats: {epoch_stats}')

        # stats['all']['lr']['0'] = self.optimizer.param_groups[0]['lr']
        # stats['all']['lr']['1'] = self.optimizer.param_groups[1]['lr']
        return stats


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    def _get_metrics(self, seg_totals):
        raise NotImplementedError

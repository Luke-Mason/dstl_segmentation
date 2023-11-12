import sys
import argparse
import torch
import os
from models.ufonet import UFONet
from models.enet import ENet
import models
from utils.ptflops import get_model_complexity_info
import json

def _get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    print('Cuda is available?: ', torch.cuda.is_available())
    print('Count of using GPUs:', torch.cuda.device_count())
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print(
            f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu

    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    print(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
    available_gpus = list(range(n_gpu))
    return device, available_gpus

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

if __name__ == '__main__':
    resolution = 256
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-g', '--gpus', default=0, type=int, help=("Number of "
                                                                "GPUs to use"))
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    device, available_gpus = _get_available_devices(args.gpus)
    model = get_instance(models, 'arch', config, 2)
    model = torch.nn.DataParallel(model, device_ids=available_gpus)
    model.to(device)

    flops, params = get_model_complexity_info(model, (3, resolution, resolution),
                                              as_strings=True,
                                              print_per_layer_stat=True,
                                              ost=ost)
    description = str(resolution) + ',' + str(flops) + ',' + str(params)
    print('Resolution: ' + str(resolution))
    print('Flops: ' + str(flops))
    print('Params: ' + str(params))

    logFileLoc = 'UFO_256_flops.txt'

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write('Resolution,Flops,Params'+'\n')
    logger.write(description + '\n')
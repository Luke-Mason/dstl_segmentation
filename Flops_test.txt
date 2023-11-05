import sys
import argparse
import torch
import os
from models.ufonet import UFONet
from models.enet import ENet
from utils.ptflops import get_model_complexity_info



pt_models = {
    'UFONet': UFONet,
    'ENet': ENet
    }

if __name__ == '__main__':
    resolution = 128
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='UFONet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()


    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        net = pt_models[args.model](num_classes=2).cuda()

        flops, params = get_model_complexity_info(net, (3, resolution, resolution),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        description = str(resolution) + ',' + str(flops) + ',' + str(params)
        print('Resolution: ' + str(resolution))
        print('Flops: ' + str(flops))
        print('Params: ' + str(params))

        logFileLoc = str(args.model) + '_flops.txt'

        if os.path.isfile(logFileLoc):
            logger = open(logFileLoc, 'a')
        else:
            logger = open(logFileLoc, 'w')
            logger.write('Resolution,Flops,Params'+'\n')
        logger.write(description + '\n')
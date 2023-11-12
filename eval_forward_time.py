import time
import torch
import torch.backends.cudnn as cudnn
import dataloaders
import models
import json
import os
from argparse import ArgumentParser
from Flops_test import _get_available_devices, get_instance


def compute_speed(model, input_size, device, iteration=1000):
    cudnn.benchmark = True
    model = model.cuda()
    model.eval()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    # torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    # torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))

    description = str(iteration) + ',' + str(input_size[3]) + ',' + str(elapsed_time) + ',' \
                + str(speed_time) + ',' + str(fps)
    logFileLoc = 'forward_time.txt'
    print(logFileLoc)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write('Iterations,Input_size, Elapsed_Time(s/100_%d iters),Speed_Time(ms/iter),FPS'% (iteration) + '\n')
    logger.write(description + '\n')

    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="256,256", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument("-g", "--gpus", default=0, type=int, help="gpu ids "
                                                                    "(default: 0)")
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    h, w = map(int, args.size.split(','))

    device, available_gpus = _get_available_devices(args.gpus)
    model = get_instance(models, 'arch', config, 2)
    model = torch.nn.DataParallel(model, device_ids=available_gpus)
    model.to(device)

    compute_speed(model, (args.batch_size, args.num_channels, h, w), device, iteration=args.iter)

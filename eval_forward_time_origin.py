import time
import torch
import torch.backends.cudnn as cudnn
import dataloaders
import models
import json
import os
from argparse import ArgumentParser



def compute_speed(model, input_size, device, iteration=1000):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
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
    #parser.add_argument('--model', type=str, default='UFONet')
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    h, w = map(int, args.size.split(','))
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    num_classes = loader.dataset.num_classes
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    compute_speed(model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)

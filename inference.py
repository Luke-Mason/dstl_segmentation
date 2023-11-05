import argparse
import scipy
import os
import numpy as np
import json
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict
from utils.metrics import eval_metrics, AverageMeter
from dataloaders.coco_cat import ID_TO_TRAINID


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')

    args = parser.parse_args()
    config = json.load(open(args.config))

    path_model = 'saved/cat_512_batch32/12-26_21-14/'
    print(path_model)
    h = 512
    w = h
    folder = 'cat'

    args.model = os.path.join(path_model,'best_model.pth')
    root ='/home/kaien125/experiments/data/coco'
    imagesSubFolder = 'images/val2017'
    labelsSubFolder = 'annotations/val2017'
    # imagesSubFolder = 'images/val2017'
    # labelsSubFolder = 'annotations/val2017'
    path = os.path.join(root,folder)
    args.images = os.path.join(path, imagesSubFolder)
    # args.images = '/home/kaien125/experiments/data/coco/street/images/test'
    args.labels = os.path.join(path, labelsSubFolder)
    # args.labels = '/home/kaien125/experiments/data/coco/street/annotations/test'
    args.extension = 'jpg'
    args.mode = 'multiscale'
    args.output = os.path.join(path_model,'outputs')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    label_ext = 'png'


    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0


    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K']
    if dataset_type == 'CityScapes': 
        scales = [1.0]
    else:
        scales = [1.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette
    #print(num_classes)

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    print("INFERENCE!!!")
    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    label_files = sorted(glob(os.path.join(args.labels, f'*.{label_ext}')))
    with torch.no_grad():
        for i in tqdm(range(len(image_files))):
            img_file = image_files[i]
            lab_file = label_files[i]
            image = Image.open(img_file).convert('RGB')
            image = np.asarray(image)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

            label = Image.open(lab_file)
            label = np.asarray(label, dtype=np.int32)

            for k, v in ID_TO_TRAINID.items():
                label[label == k] = v

            #label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            origin_w, origin_h = label.shape[0], label.shape[1]

            input = normalize(to_tensor(image)).unsqueeze(0)

            # print(np.unique(label))

            # if args.mode == 'multiscale':
            #     prediction = multi_scale_predict(model, input, scales, num_classes, device)
            # elif args.mode == 'sliding':
            #     prediction = sliding_predict(model, input, num_classes)
            # else:
            #     prediction = model(input.to(device))
                #prediction = prediction.squeeze(0).cpu().numpy()
            #prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

            output = model(input.to(device))
            output = F.interpolate(output, size=(origin_w, origin_h), mode = 'bicubic', align_corners = True)

            prediction = output.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            # print(type(prediction))
            # print((prediction.shape))
            image = Image.fromarray(image.astype('uint8'), 'RGB')

            save_images(image, prediction, args.output, img_file, palette)

            label = torch.from_numpy(label)
            label = label.to(device)
            # print(prediction.shape()[2:])
            # print(label.shape()[0:])
            # assert prediction.shape()[2:] == label.shape()[0:]
            correct, labeled, inter, union = eval_metrics(output, label, num_classes)
            # print("correct" + str(correct))
            # print('labeled'+ str(labeled))
            #_update_seg_metrics(*seg_metrics)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

            #pixAcc, mIoU, _ = _get_seg_metrics().values()
        print('Pixel_Accuracy: ' + str(pixAcc))
        print('Mean_IoU: ' + str(mIoU))

        class_Iou = []
        classNum_list = []
        for i in range(len(IoU)):
            classNum = 'Class_' + str(i) + '_IoU'
            classNum_list.append(classNum)
            class_Iou.append(str(IoU[i]))
            print(classNum + ': ' + str(IoU[i]))

        description = str(folder) + ',' + str(h) + ',' + str(pixAcc) + ',' + str(mIoU) + ',' + ",".join(class_Iou)
        logFileLoc = os.path.join(path_model, 'test_result.txt')
        # logFileLoc_all = os.path.join('/home/kaien125/experiments/code/ufonet_experiment', folder + '_single_class_iou_inference_result.txt')
        logFileLoc_all = os.path.join('/home/kaien125/experiments/code/ufonet_experiment',
                                      'single_class_iou_inference_result.txt')
        print(logFileLoc)
        if os.path.isfile(logFileLoc):
            logger = open(logFileLoc, 'a')
        else:
            logger = open(logFileLoc, 'w')
            logger.write('Subset,Resolution,Pixel_Accuracy,Mean_IoU,' + ",".join(classNum_list) + '\n')
        logger.write(description + '\n')

        if os.path.isfile(logFileLoc_all):
            logger_all = open(logFileLoc_all, 'a')
        else:
            logger_all = open(logFileLoc_all, 'w')
            logger_all.write('Subset,Resolution,Pixel_Accuracy,Mean_IoU,' + ",".join(classNum_list) + '\n')
        logger_all.write(description + '\n')




def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='saved/UFONet_coco_wildAnimal_512_batch8_4class/07-04_22-40/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default='/home/kaien125/experiments/data/coco/hairdrier/images/val2017', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()

    return args




# def _update_seg_metrics(correct, labeled, inter, union):
#     total_correct += correct
#     total_label += labeled
#     total_inter += inter
#     total_union += union
#
# def _get_seg_metrics():
#     pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
#     IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
#     mIoU = IoU.mean()
#     return {
#         "Pixel_Accuracy": np.round(pixAcc, 3),
#         "Mean_IoU": np.round(mIoU, 3),
#         "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
#     }

if __name__ == '__main__':
    main()

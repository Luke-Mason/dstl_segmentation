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
from utils.palette import COCO_palette
from scipy.ndimage.measurements import label as labeling
import time


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
    output_im = Image.new('RGB', (w*2, h))
    output_im.paste(image, (0,0))
    output_im.paste(colorized_mask, (w,0))
    output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    mask_img = Image.fromarray(mask, 'L')
    mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')

    args = parser.parse_args()
    config = json.load(open(args.config))

    path_model = 'saved/cat_128_batch32/12-12_12-01'
    h = 128
    w = h
    folder = 'cat'
    # dict_label = {1:'elephant', 2:'zebra', 3:'girrafe'}
    # dict_color = {1:3, 2:6, 3:9}

    dict_label = {1:'cat'}
    dict_color = {1:6}

    args.model = os.path.join(path_model,'best_model.pth')
    root ='/home/kaien125/experiments/data/coco'
    # imagesSubFolder = 'images/test'
    # labelsSubFolder = 'annotations/test'
    imagesSubFolder = 'images/val2017'
    labelsSubFolder = 'annotations/val2017'
    # path = os.path.join(root,folder)
    args.images = os.path.join(root, imagesSubFolder)
    # args.images = '/home/kaien125/experiments/data/coco/street/images/test'
    args.labels = os.path.join(root, labelsSubFolder)
    # args.labels = '/home/kaien125/experiments/data/coco/street/annotations/test'
    args.extension = 'jpg'
    args.mode = 'multiscale'
    args.output = os.path.join(path_model,'outputs')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    label_ext = 'png'

    bbox_path = os.path.join(path_model,'bbox_outputs')
    Path(bbox_path).mkdir(parents=True, exist_ok=True)

    # bbox_txt_path = os.path.join(path_model, 'bbox_outputs/bbox_txt/')
    bbox_txt_path = os.path.join(path_model, 'bbox_outputs/all_bbox_val2017_txt/')
    Path(bbox_txt_path).mkdir(parents=True, exist_ok=True)


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

    if not os.path.exists(bbox_path + '/all/'):
        os.makedirs(bbox_path + '/all/')

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    label_files = sorted(glob(os.path.join(args.labels, f'*.{label_ext}')))
    with torch.no_grad():
        for i in tqdm(range(len(image_files))):
            img_file = image_files[i]
            lab_file = label_files[i]
            image_origin = Image.open(img_file).convert('RGB')

            image_origin = np.asarray(image_origin)

            image = cv2.resize(image_origin, (w, h), interpolation=cv2.INTER_LINEAR)

            label = Image.open(lab_file)
            label = np.asarray(label, dtype=np.int32)

            for k, v in ID_TO_TRAINID.items():
                label[label == k] = v

            origin_w, origin_h = label.shape[0], label.shape[1]

            input = normalize(to_tensor(image)).unsqueeze(0)

            output = model(input.to(device))
            output = F.interpolate(output, size=(origin_w, origin_h), mode = 'bicubic', align_corners = True)

            prediction = output.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            # print(type(prediction))
            # print((prediction.shape))
            uni_values = np.unique(prediction)

            font = cv2.FONT_HERSHEY_SIMPLEX

            bbox_image_name = os.path.basename(img_file).split('.')[0]
            with open(bbox_txt_path + bbox_image_name + ".txt", "w") as new_f:
                if len(uni_values) > 1:
                    label_list = uni_values[1:]
                    # choose bbox or bbox_large for bounding box method
                    bbox_list = bbox(prediction, label_list)
                    for draw_box in bbox_list:
                        label = draw_box[0]
                        palette_index = dict_color.get(label)
                        cmin, rmin, cmax, rmax  = draw_box[1]
                        cv2.rectangle(image_origin, (cmin, rmin), (cmax, rmax), (COCO_palette[palette_index], COCO_palette[palette_index + 1], COCO_palette[palette_index + 2]), 2)
                        text = dict_label.get(label)
                        # Meaning of 7 parameters (picture, text information, placement position, font, font size, font color, thickness)
                        cv2.putText(image_origin, text, (cmin + 5, rmax - 5), font, 0.7, (COCO_palette[palette_index], COCO_palette[palette_index + 1], COCO_palette[palette_index + 2]), 2)
                        new_f.write("%s %s %s %s %s %s\n" % (text, 1, cmin, rmin, cmax, rmax))

                    image_name = bbox_path + '/all/' + bbox_image_name + '.jpg'
                    cv2.imwrite(image_name, cv2.cvtColor(image_origin, cv2.COLOR_RGB2BGR))

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


# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(img, label_list):
    output_list = []
    for label in label_list:
        result = img.copy()
        result[result != label] = 0
        result = np.uint8(result)
        # threshold
        thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)[1]
        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if w > 20 and h > 20:
                output = (x, y, x + w, y + h)
                output_list.append([label, output])
    return output_list

#
def bbox_large(img, label_list):
    output_list = []
    for label in label_list:
        result = img.copy()
        result = np.where(result == label)
        output = (np.min(result[1]), np.min(result[0]), np.max(result[1]), np.max(result[0]))
        output_list.append([label, output])
    return output_list


if __name__ == '__main__':
    main()

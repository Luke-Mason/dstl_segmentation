import argparse
import scipy
import os
import time
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

    #-------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    #-------------------------------------------------------------------------#
    # video_path = 0
    video_path = '/home/kaien125/experiments/data/zebra.mp4'
    video_fps = 25.0

    path_model = 'saved/wildlife_128_4class/07-17_11-16'
    video_save_path = os.path.join(path_model,'video_bbox_outputs')
    video_name = 'zebra.mp4'
    Path(video_save_path).mkdir(parents=True, exist_ok=True)
    video_save_path = os.path.join(video_save_path, video_name)


    h = 128
    w = h
    folder = 'wildlife'
    dict_label = {1:'elephant', 2:'zebra', 3:'giraffe'}
    dict_color = {1:3, 2:6, 3:9}


    args.model = os.path.join(path_model,'best_model.pth')
    # root ='/home/kaien125/experiments/data/coco'
    # imagesSubFolder = 'images/test'
    # labelsSubFolder = 'annotations/test'
    # path = os.path.join(root,folder)
    # args.images = os.path.join(path, imagesSubFolder)
    # args.images = '/home/kaien125/experiments/data/coco/street/images/test'
    # args.labels = os.path.join(path, labelsSubFolder)
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

    capture = cv2.VideoCapture(video_path)

    if video_save_path != "":
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    fps = 0.0
    while (capture.isOpened()):
        with torch.no_grad():
            t1 = time.time()
            # 读取某一帧
            ref, frame_origin = capture.read()
            if ref:
                # print(frame_origin.shape)
                #
                # origin_w, origin_h = frame_origin.shape[0], frame_origin.shape[1]
                frame = cv2.resize(frame_origin, (w, h), interpolation=cv2.INTER_LINEAR)
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                # frame = Image.fromarray(np.uint8(frame))
                frame = normalize(to_tensor(frame)).unsqueeze(0)
                # 进行检测
                output_frame = model(frame.to(device))
                output_frame = F.interpolate(output_frame, size=(size[1],size[0]), mode='bicubic', align_corners=True)

                prediction_frame = output_frame.squeeze(0).cpu().numpy()
                prediction_frame = F.softmax(torch.from_numpy(prediction_frame), dim=0).argmax(0).cpu().numpy()
                # print(type(prediction))
                # print((prediction.shape))
                uni_values = np.unique(prediction_frame)
                print(list(dict_label.get(i) for i in uni_values[1:]))

                if len(uni_values) > 1:
                    label_list = uni_values[1:]
                    for label in label_list:
                        palette_index = dict_color.get(label)
                        rmin, rmax, cmin, cmax = bbox(prediction_frame, label)

                        cv2.rectangle(frame_origin, (cmin, rmin), (cmax, rmax), (
                        COCO_palette[palette_index + 2], COCO_palette[palette_index + 1], COCO_palette[palette_index]), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = dict_label.get(label)
                        # Meaning of 7 parameters (picture, text information, placement position, font, font size, font color, thickness)
                        cv2.putText(frame_origin, text, (cmin + 5, rmax - 5), font, 0.7, (
                        COCO_palette[palette_index + 2], COCO_palette[palette_index + 1], COCO_palette[palette_index]), 2)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))

                frame_origin = cv2.putText(frame_origin, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # RGBtoBGR满足opencv显示格式
                # frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_RGB2BGR)
                # uni_values1 = np.unique(frame_origin)
                # print(uni_values1)
                # print(type(frame_origin))
                # print(type(prediction_frame))
                # cv2.imshow("video", prediction_frame.astype('uint8'))
                cv2.imshow("video", frame_origin)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame_origin)

                if c == 27:
                    capture.release()
                    break
            else:
                break
    capture.release()
    out.release()
    cv2.destroyAllWindows()

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
def bbox(img, label):
    a = np.where(img == label)
    rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return rmin, rmax, cmin, cmax

#
#
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

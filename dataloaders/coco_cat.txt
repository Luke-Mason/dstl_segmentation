# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from utils import palette
import torch
import os
import cv2

ID_TO_TRAINID = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0,
                 11:0, 12:0, 13:0, 14:0, 15:0, 16:1, 17:0, 18:0, 19:0,
                 20:0, 21:0, 22:0, 23:0, 24:0, 25:0, 26:0, 27:0, 28:0,
                 29:0, 30:0, 31:0, 32:0, 33:0, 34:0, 35:0, 36:0, 37:0,
                 38:0, 39:0, 40:0, 41:0, 42:0, 43:0, 44:0, 45:0, 46:0,
                 47:0, 48:0, 49:0, 50:0, 51:0, 52:0, 53:0, 54:0, 55:0,
                 56:0, 57:0, 58:0, 59:0, 60:0, 61:0, 62:0, 63:0, 64:0,
                 65:0, 66:0, 67:0, 68:0, 69:0, 70:0, 71:0, 72:0, 73:0,
                 74:0, 75:0, 76:0, 77:0, 78:0, 79:0, 80:0, 81:0, 82:0,
                 83:0, 84:0, 85:0, 86:0, 87:0, 88:0, 89:0, 90:0, 91:0,
                 92:0, 93:0, 94:0, 95:0, 96:0, 97:0, 98:0, 99:0, 100:0,
                 101:0, 102:0, 103:0, 104:0, 105:0, 106:0, 107:0, 108:0,
                 109:0, 110:0, 111:0, 112:0, 113:0, 114:0, 115:0, 116:0,
                 117:0, 118:0, 119:0, 120:0, 121:0, 122:0, 123:0, 124:0,
                 125:0, 126:0, 127:0, 128:0, 129:0, 130:0, 131:0, 132:0,
                 133:0, 134:0, 135:0, 136:0, 137:0, 138:0, 139:0, 140:0,
                 141:0, 142:0, 143:0, 144:0, 145:0, 146:0, 147:0, 148:0,
                 149:0, 150:0, 151:0, 152:0, 153:0, 154:0, 155:0, 156:0,
                 157:0, 158:0, 159:0, 160:0, 161:0, 162:0, 163:0, 164:0,
                 165:0, 166:0, 167:0, 168:0, 169:0, 170:0, 171:0, 172:0,
                 173:0, 174:0, 175:0, 176:0, 177:0, 178:0, 179:0, 180:0,
                 181:0, 182:0, 255:0}

# ID_TO_TRAINID = {0: 0, 1: 0, 2: 0,
#                     3: 0, 4: 0, 5: 0, 6: 0,
#                     7: 0, 8: 0, 9: 0, 10: 1, 11: 0, 12: 0, 13: 1,
#                     14: 0, 15: 0, 16: 0, 17: 1,
#                     18: 0, 19: 0, 20: 0}

# class CocoStuff10k(BaseDataSet):
#     def __init__(self, warp_image = True, **kwargs):
#         self.warp_image = warp_image
#         self.num_classes = 2
#         self.palette = palette.COCO_palette
#         self.id_to_trainId = ID_TO_TRAINID
#         super(CocoStuff10k, self).__init__(**kwargs)
#
#     def _set_files(self):
#         if self.split in ['train', 'test', 'all']:
#             file_list = os.path.join(self.root, 'imageLists', self.split + '.txt')
#             self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]
#         else: raise ValueError(f"Invalid split name {self.split} choose one of [train, test, all]")
#
#     def _load_data(self, index):
#         image_id = self.files[index]
#         image_path = os.path.join(self.root, 'images', image_id + '.jpg')
#         label_path = os.path.join(self.root, 'annotations', image_id + '.png')
#         image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
#         label = np.asarray(Image.open(label_path), dtype=np.int32)
#         if self.warp_image:
#             image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
#             label = np.asarray(Image.fromarray(label).resize((513, 513), resample=Image.NEAREST))
#         for k, v in self.id_to_trainId.items():
#             label[label == k] = v
#         return image, label, image_id

class CocoStuff164k(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.COCO_palette
        self.id_to_trainId = ID_TO_TRAINID
        super(CocoStuff164k, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'val','train2017', 'val2017']:
            file_list = sorted(glob(os.path.join(self.root, 'images', self.split + '/*.jpg')))
            self.files = [os.path.basename(f).split('.')[0] for f in file_list]
        else: raise ValueError(f"Invalid split name {self.split}, either train2017 or val2017")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', self.split, image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', self.split, image_id + '.png')
        image_o = Image.open(image_path).convert('RGB')

        # #mono channel
        # image_o = cv2.imread(image_path)
        # gray = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)
        # image_o = np.zeros_like(image_o)
        # image_o[:, :, 0] = gray
        # image_o[:, :, 1] = gray
        # image_o[:, :, 2] = gray


        image = np.asarray(image_o, dtype=np.float32)
        label_o = Image.open(label_path)
        label = np.asarray(label_o, dtype=np.int32)
        # print(label.shape[0])
        # print(label.shape[1])


        for k, v in self.id_to_trainId.items():
            label[label == k] = v


        #print(np.unique(label))

        unique, counts = np.unique(label, return_counts=True)
        pixel_count = dict(zip(unique, counts))
        # commment out when not getting subset
        # if (1 in label) and (pixel_count[1]/label.shape[0]/label.shape[1] > 0.0):
        #     image_path = os.path.join(self.root, 'fire_hydrant','images', self.split, image_id + '.jpg')
        #     print(image_path)
        #     label_path = os.path.join(self.root, 'fire_hydrant','annotations', self.split, image_id + '.png')
        #     image_o.save(image_path)
        #     label_o.save(label_path)


        return image, label, image_id

def get_parent_class(value, dictionary):
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res

class COCO(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, partition = 'CocoStuff164k',
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if partition == 'CocoStuff10k': self.dataset = CocoStuff10k(**kwargs)
        elif partition == 'CocoStuff164k': self.dataset = CocoStuff164k(**kwargs)
        else: raise ValueError(f"Please choose either CocoStuff10k / CocoStuff164k")

        super(COCO, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


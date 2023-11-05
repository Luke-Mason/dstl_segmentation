
path = '/home/kaien125/experiments/data/coco/wildAnimal/images/train2017/000000546869.jpg'
import cv2
import numpy as np
from PIL import Image
import os
import glob
from utils.palette import COCO_palette as palette
from utils.helpers import colorize_mask
import pandas as pd
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

# image_o = Image.open(path).convert('L')
# image_o = Image.open(path).convert('RGB')
# image_o.show()
# num_channel = len(image_o.split())
# print(num_channel)

# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img2 = np.zeros_like(img)
# img2[:,:,0] = gray
# img2[:,:,1] = gray
# img2[:,:,2] = gray
#
# #cv2.circle(img2, (10,10), 5, (255,255,0))
# cv2.imshow("colour again", img2)
# cv2.waitKey()
#
# img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#
# print('Original Dimensions : ', img.shape)
#
# dim = (512, 512)
#
# # resize image
# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (450,50)
# fontScale              = 1.8
# fontColor              = (0,0,0)
# lineType               = 6
#
# cv2.putText(resized,'A',
#     bottomLeftCornerOfText,
#     font,
#     fontScale,
#     fontColor,
#     lineType)
#
# print('Resized Dimensions : ', resized.shape)
# #cv2.imwrite('/home/kaien125/Pictures/Figure_A.jpg',resized)
#
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
id_to_trainId = {0:0, 1:1, 2:0, 3:0, 4:0, 5:2, 6:0, 7:3, 8:0, 9:0, 10:0,
                 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0,
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

root_label = '/home/kaien125/experiments/paper/predictions/label'

#root_label = '/home/kaien125/experiments/data/coco/wildlife/annotations'
root_lab_train = os.path.join(root_label, 'vehicle_4class')
glob_labels = glob.glob(os.path.join(root_lab_train, '*.png'))




for g_lab in glob_labels:
    if g_lab.endswith(".png"):
        label_o = Image.open(g_lab)
        label = np.asarray(label_o, dtype=np.int32)
        for k, v in id_to_trainId.items():
            label[label == k] = v
        # label = colorize_mask(label, palette)
        # print(label.size)
        # label = label.convert('RGB')
        # print(type(label))
        # print(label.size)
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)

        pred_r = np.zeros_like(label)
        pred_g = np.zeros_like(label)
        pred_b = np.zeros_like(label)

        for x in range(0, label.shape[0]):
            for y in range(0, label.shape[1]):
                pred_r[x][y] = palette[label[x][y] * 3]
                pred_g[x][y] = palette[label[x][y] * 3 + 1]
                pred_b[x][y] = palette[label[x][y] * 3 + 2]


        label = cv2.merge((pred_r, pred_g, pred_b))

        label = Image.fromarray(np.uint8(label)).convert('RGB')

        label.save(g_lab)



image_list = []
label_list = []
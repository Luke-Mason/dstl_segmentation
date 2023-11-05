import cv2
from PIL import Image
import os
import numpy as np

mask_path = '/home/kaien125/experiments/paper/predictions/city1024/munster_000003_000019_leftImg8bit_1024.png'
img_path = '/home/kaien125/experiments/data/cityscapes/leftImg8bit/val/munster/munster_000003_000019_leftImg8bit.png'
# image = Image.open(img_path).convert('RGB')
# mask = Image.open(mask_path).convert('RGB')
image = cv2.imread(img_path)

image = cv2.resize(image, (2048,1024), interpolation = cv2.INTER_AREA)
mask = cv2.imread(mask_path)
mask = cv2.resize(mask, (2048,1024), interpolation = cv2.INTER_AREA)

# use `addWeighted` to blend the two images
out = cv2.addWeighted(image, 0.8, mask, 0.5,0)

filename = 'munster_000003_000019_leftImg8bit_maskOverlay.png'
cv2.imwrite(filename, out)
cv2.imshow('image',out.astype(np.uint8))
cv2.waitKey(0)

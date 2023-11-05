import cv2
from PIL import Image
import os
import numpy as np

mask_path = '/home/kaien125/experiments/paper/predictions/256/wildlife/000000295887_wildlife_256.png'
img_path = '/home/kaien125/experiments/paper/predictions/image/wildlife/000000295887_wildlife_image.jpg'
# image = Image.open(img_path).convert('RGB')
# mask = Image.open(mask_path).convert('RGB')
image = cv2.imread(img_path)
mask = cv2.imread(mask_path)/255
mask = mask.astype(int)
print(type(mask))
show = image*mask
print(np.unique(mask))
filename = '000000295887_wildlife_rpn.jpg'
cv2.imwrite(filename, show)
cv2.imshow('image',show.astype(np.uint8))
cv2.waitKey(0)

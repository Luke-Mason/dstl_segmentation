import cv2
from PIL import Image
import os
import numpy as np

mask_path = '/home/kaien125/experiments/data/coco/annotations/train2017/000000000836.png'
mask = cv2.imread(mask_path)
print(np.unique(mask))



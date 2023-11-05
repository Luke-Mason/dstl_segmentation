import cv2
import numpy as np
from PIL import Image
import os
import glob

root_label = '/home/kaien125/experiments/paper/predictions'

#root_label = '/home/kaien125/experiments/data/coco/wildlife/annotations'
root_lab_train = os.path.join(root_label, 'test')
glob_labels = glob.glob(os.path.join(root_lab_train, '*.png'))



def bbox1(img, label):
    a = np.where(img == label)
    rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return rmin, rmax, cmin, cmax


# for g_lab in glob_labels:
#     if g_lab.endswith(".png"):
#         file_name = os.path.basename(g_lab)
#         img = cv2.imread(g_lab)
#         label_list = np.unique(img)[1:]
#         for label in label_list:
#             rmin, rmax, cmin, cmax = bbox1(img, label)
#             print(rmin)
#             cv2.rectangle(img, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
#         cv2.imwrite('/home/kaien125/experiments/paper/predictions/test/output/'+file_name, img)

for g_lab in glob_labels:
    if g_lab.endswith(".png"):
        file_name = os.path.basename(g_lab)
        img = cv2.imread(g_lab)
        print(type(img))
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
    cv2.imwrite('/home/kaien125/experiments/paper/predictions/test/output/' + file_name, result)
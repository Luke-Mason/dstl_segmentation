import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

TRAIN_ANNOTATIONS_PATH = "/home/kaien125/experiments/data/coco/annotations/annotations_trainval2017/annotations/instances_train2017.json"

coco_annotation = COCO(TRAIN_ANNOTATIONS_PATH)



# # Category IDs.
cat_ids = coco_annotation.getCatIds()

cats = coco_annotation.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats]

for i in range(len(cat_names)):
    print(str(i) + " " + cat_names[i])
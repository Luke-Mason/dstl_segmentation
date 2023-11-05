import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import glob
from tqdm import tqdm


TRAIN_ANNOTATIONS_PATH = "/home/kaien125/experiments/data/coco/annotations/annotations_trainval2017/annotations/instances_train2017.json"
VAL_ANNOTATIONS_PATH = "/home/kaien125/experiments/data/coco/annotations/annotations_trainval2017/annotations/instances_val2017.json"

coco_annotation = COCO(TRAIN_ANNOTATIONS_PATH)
coco_annotation_val = COCO(VAL_ANNOTATIONS_PATH)


# # Category IDs.
cat_ids = coco_annotation.getCatIds()
# print(f"Number of Unique Categories: {len(cat_ids)}")
# print("Category IDs:")
# print(cat_ids)  # The IDs are not necessarily consecutive.

# All categories.
cats = coco_annotation.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats]

# # Category Name -> Category ID.
# query_name = cat_names[6]
# query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
# print("Category Name -> ID:")
# print(f"Category Name: {query_name}, Category ID: {query_id}")
#
# # Get the ID of all the images containing the object of the category.
# img_ids = coco_annotation.getImgIds(catIds=[query_id])
# print(f"Number of Images Containing {query_name}: {len(img_ids)}")
#
# # Pick one image.
# img_id = img_ids[2]
# print(img_id)
# img_info = coco_annotation.loadImgs([110])[0]
# img_file_name = img_info["file_name"]
# img_url = img_info["coco_url"]
# print(
#     f"Image ID: {img_id}, File Name: {img_file_name}, Image URL: {img_url}"
# )
#
# Get all the annotations for the specified image.
# ann_ids = coco_annotation.getAnnIds(imgIds=[397], iscrowd=None)
# anns = coco_annotation.loadAnns(ann_ids)
# print(f"Annotations for Image ID {397}:")
# print(anns)

# for i in range(len(anns)):
#     if anns[i]['category_id'] <= 80:
#         cls = cat_names[anns[i]['category_id']-1]
#         # if cls in cls_list:
#         print(cls)
#
# for i in range(len(anns)):
#     if anns[i]['category_id'] == 3:
#         print(anns[i]['bbox'][0])




# # All categories.
# cats = coco_annotation.loadCats(cat_ids)
# print(type(cats))
# cat_names = [cat["name"] for cat in cats]
# print("Categories Names:")
# print(cat_names)
# print(len(cat_names))

root = '/home/kaien125/experiments/data/coco'
folder = 'stop_sign'
cls_list = ['stop sign']

# image_folder = os.path.join(root, folder + '/images/test')
# output_folder = os.path.join(root, folder + '/annotations/bbox_test/')

# image_folder = os.path.join(root, folder + '/images/val2017')
# output_folder = os.path.join(root, folder + '/annotations/bbox_val2017/')

image_folder = os.path.join(root + '/images/val2017')
print(image_folder)
output_folder = os.path.join(root, folder + '/annotations/all_bbox_val2017/')

image_ids = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img in tqdm(image_ids):
    img_file_name = os.path.basename(img)
    img_id_0 = os.path.splitext(img_file_name)[0]
    img_id = img_id_0.lstrip("0")
    # print(img_id)
    # img_info = coco_annotation.loadImgs([img_id])[0]
    # print(img_id)
    # print(type(int(img_id)))
    # print(type(510103))

    try:
        ann_ids = coco_annotation.getAnnIds(imgIds=int(img_id), iscrowd=None)
        print(ann_ids)
        anns = coco_annotation.loadAnns(ann_ids)

        img_info = coco_annotation.loadImgs([int(img_id)])[0]
        img_file_name = img_info["file_name"]
        print(img_file_name)
    except:
        print("image is not in training data")

    try:
        ann_ids_val = coco_annotation_val.getAnnIds(imgIds=int(img_id), iscrowd=None)
        anns_val = coco_annotation_val.loadAnns(ann_ids_val)

        img_info_val = coco_annotation_val.loadImgs(int(img_id))[0]
        img_file_name_val = img_info_val["file_name"]
        print(img_file_name_val)
    except:
        print("image is not in val data")

    anns = anns + anns_val
    # print(f"Annotations for Image ID {img_id}:")

    # # All categories.
    # cat_ids = coco_annotation.getCatIds('person')
    # print(cat_ids)
    # cats = coco_annotation.loadCats(cat_ids)
    # print(type(cats))
    # cat_names = [cat["name"] for cat in cats]
    # print("Categories Names:")
    # print(cat_names)


    with open(output_folder +img_id_0+".txt", "w") as new_f:
        for i in range(len(anns)):
            if anns[i]['category_id'] <= 80:
                cls = cat_names[anns[i]['category_id']-2]
                print(cls + ' ' + str(anns[i]['category_id']-2))
                if cls in cls_list:
                    _cls = cls.split()
                    if len(_cls) > 1:
                        cls = _cls[0] + '_' + _cls[1]
                    left, top, right, bottom = anns[i]['bbox'][0],\
                                           anns[i]['bbox'][1],\
                                           anns[i]['bbox'][0] + anns[i]['bbox'][2],\
                                           anns[i]['bbox'][1] + anns[i]['bbox'][3]

                    new_f.write("%s %s %s %s %s\n" % (cls, left, top, right, bottom))





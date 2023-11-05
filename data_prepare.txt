import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

root_image = '/home/kaien125/experiments/data/coco/banana/images'
root_img_train = os.path.join(root_image, 'train2017')
glob_images = glob.glob(os.path.join(root_img_train, '*.jpg'))

root_label = '/home/kaien125/experiments/data/coco/banana/annotations'
root_lab_train = os.path.join(root_label, 'train2017')
glob_labels = glob.glob(os.path.join(root_lab_train, '*.png'))

image_list = []
label_list = []

write_image = open(root_image+'_list.txt','w')
for g_img in glob_images:
    # g_img = g_img.replace(root_img_train + '/', '')
    # g_img = g_img.replace('.jpg', '')
    image_list.append(g_img)
    write_image.write(g_img+'\n')
write_image.close()

write_label = open(root_label+'_list.txt','w')
for g_lab in glob_labels:
    # g_lab = g_lab.replace(root_lab_train + '/', '')
    # g_lab = g_lab.replace('.png', '')
    label_list.append(g_lab)
    write_label.write(g_lab+'\n')
write_label.close()

print('total: ' + str(len(image_list)))

assert len(image_list) == len(label_list)

image_list.sort()
label_list.sort()

data = pd.DataFrame(
    {'images': image_list,
     'labels': label_list
    })

train, test, val = np.split(data.sample(frac=1, random_state=42),
                       [int(.7*len(data)), int(.9*len(data))])

print('train: ' + str(len(train)))
print('val: ' + str(len(val)))
print('test: ' + str(len(test)))

df_list = [train, val, test]

# create folders
folder_list = ['train', 'val', 'test']
for folder in folder_list:
    path_image = os.path.join(root_image, folder)
    path_label = os.path.join(root_label, folder)
    Path(path_image).mkdir(parents=True, exist_ok=True)
    Path(path_label).mkdir(parents=True, exist_ok=True)

# copy files to destinations
for index in range(len(df_list)):
    for row_index, row in tqdm(df_list[index].iterrows()):
        dest_img = row['images'].replace('train2017', folder_list[index])
        dest_lab = row['labels'].replace('train2017', folder_list[index])

        copyfile(row['images'], dest_img)
        copyfile(row['labels'], dest_lab)



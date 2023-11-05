import os

import numpy as np
import pandas as pd
import rasterio

means = []
stds = []
num_bands = 17

# File to load metadata for the training data set
# file_path = dataset_path + 'dstl_data.csv'
# root_path = '/mnt/e/ML_DATA/DSTL/dstl-satellite-imagery-feature-detection/'
root_path = '/opt/home/s3630120/dstl-satellite-imagery-feature-detection/'
dataset_path = root_path + 'cached/'
file_path = root_path + 'train_wkt_v4.csv/train_wkt_v4.csv'

df = pd.read_csv(file_path)
ids = df['ImageId'].unique().tolist()

mean_data_list = dict({bnd: (0, 0) for bnd in range(num_bands)})
data_list = dict({bnd: [] for bnd in range(num_bands)})

for img_id in ids:
    print(img_id)
    path = dataset_path + f'{img_id}_interp_4.tif'
    if path is None or not os.path.exists(path):
        print(f"Could not find file for image_id: {img_id}")
        continue
    with rasterio.open(path) as src:
        for bnd in range(num_bands):
            data = src.read(bnd + 1)
            total, count = mean_data_list[bnd]
            mean_data_list[bnd] = (
                total + np.sum(data), count + (data.shape[0] * data.shape[1]))

for bnd in range(num_bands):
    print(
        f"Mean for Band {bnd + 1}: {mean_data_list[bnd][0] / mean_data_list[bnd][1]}")

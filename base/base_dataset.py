import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage

class BaseDataSet(Dataset):
    def __init__(self, root, mean, std, base_size=None, augment=True,
                 crop_size=0, scale=False, flip=False, rotate=False,
                 blur=False, return_id=False):
        self.root = root
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self._set_files()
        self.to_tensor = transforms.ToTensor()

        # Optionally normalize, does require dataset class to override __item__.
        if len(mean) != 0 and len(std) != 0:
            self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _augmentation(self, image, label):
        c, h, w = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            h = self.base_size
            w = self.base_size
            image = cv2.resize(image.transpose((1,2,0)), (w, h), interpolation=cv2.INTER_LINEAR).transpose((2,0,1))
            label = cv2.resize(label.transpose((1,2,0)), (w, h), interpolation=cv2.INTER_NEAREST).transpose((2,0,1))

        c, h, w = image.shape  # Assuming 'image' is a 3D array with dimensions (c, h, w)

        # Rotate the image and label with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-20, 20)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate the image
            image_rotated = np.zeros_like(image)
            for channel in range(c):
                image_rotated[channel] = cv2.warpAffine(image[channel],
                                                        rot_matrix, (w, h),
                                                        flags=cv2.INTER_LINEAR)

            # Rotate the label
            label_rotated = np.zeros_like(label)
            for channel in range(label.shape[0]):
                label_rotated[channel] = cv2.warpAffine(label[channel],
                                                        rot_matrix,
                                                        (label.shape[2],
                                                         label.shape[1]),
                                                        flags=cv2.INTER_NEAREST)

            image = image_rotated
            label = label_rotated

        # Padding to return the correct crop size
        if self.crop_size != 0:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,
            }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

            # Cropping
            _, h, w = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[:, start_h:end_h, start_w:end_w]
            label = label[:, start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=2).copy()

        # Gaussian Blur (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random() * 1.5
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize

            # Apply Gaussian blur to each channel
            for channel in range(c):
                image[channel] = cv2.GaussianBlur(image[channel],
                                                  (ksize, ksize), sigmaX=sigma,
                                                  sigmaY=sigma,
                                                  borderType=cv2.BORDER_REFLECT_101)

        # print("image.shape: ", image.shape)
        # print("label.shape: ", label.shape)
        return image, label
        
    def __len__(self):
        print("LEN: ", len(self.files))
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


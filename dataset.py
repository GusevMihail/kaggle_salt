import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import imageio
import cv2
import albumentations as albu
from albumentations.torch import ToTensor

from torch.utils import data


class TGSSaltDataset(data.Dataset):

    def __init__(self, root_path, file_list, transforms=None):
        self.root_path = root_path
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        mask_folder = os.path.join(self.root_path, "masks")
        image_path = os.path.join(image_folder, file_id + ".png")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def show_samples(self, n_col, n_row, random=True):
        fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
        i = 0
        for row in range(n_row):
            for col in range(n_col):
                ax = axs[row, col]
                if random:  # random samples
                    image, mask = self[np.random.randint(0, len(self))]
                else:  # first N samples
                    image, mask = self[i]
                    i += 1

                ax.imshow(np.array(image, dtype=np.uint8))
                ax.imshow(np.array(mask, dtype=np.uint8), cmap='seismic', alpha=0.2)
                ax.grid(False)
                ax.axis('off')


train_df = pd.read_csv('./data/train.csv')
depths_df = pd.read_csv('./data/depths.csv')
subm_df = pd.read_csv('./data/sample_submission.csv')
train_path = r'./data/train/'


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


transforms = compose([pre_transforms(image_size=96)])

dataset = TGSSaltDataset(train_path, list(train_df.id.values[:25]), transforms=transforms)

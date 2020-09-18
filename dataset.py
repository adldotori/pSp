from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
import os


class AIDataset(Dataset):
    def __init__(self, name, root_path, resolution):
        self.name = name
        self.root_path = root_path
        self.resolution = resolution

        self.transform_4ch = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5), inplace=True),
        ])

        self.transform_3ch = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.transform_1ch = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

        self.image_dir_lists = glob.glob(os.path.join(root_path, '*'))
        self.seg_dir_lists = glob.glob(os.path.join('distort', '*', '*'))
        self.length = len(self.image_dir_lists)

        print(f'{self.name} : {self.length}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):        
        # random flip
        # random_flip = random.randint(0, 1) # random face flip

        img_path = self.image_dir_lists[index]
        file_data = Image.open(img_path)
        shape = np.array(file_data).shape
        if shape[-1] == 4:
            image = self.transform_4ch(file_data)
        elif shape[-1] == 3:
            image = self.transform_3ch(file_data)
        
        random_seg = random.randint(0, len(self.seg_dir_lists)-1)
        seg = Image.open(self.seg_dir_lists[random_seg])
        seg = seg.resize((1024,1024))
        seg = np.array(seg)
        seg[seg>0] = 1
        seg = np.stack([seg, seg, seg], axis=0)
        seg = torch.tensor(seg)
        return image, image * seg

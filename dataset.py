from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from torchvision import transforms
import glob
import os

class MultiResolutionDataset(Dataset):
    def __init__(self, name, root_path, resolution, domain, output_type_lst):
        self.name = name
        self.root_path = root_path
        self.resolution = resolution
        self.domain = domain
        self.output_type_lst = output_type_lst

        # pytorch 1.4 dose not support
        # transforms.Normalize(0.5, 0.5, inplace=True)
        # for various channels
        # veresion over 1.5 supports it

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
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ])
        
        if domain == 0:
            dir_lst = ['men-clothing-shirts', 'mens-clothing-pullover-cardigans', 'mens-clothing-shirts',
                                    'mens-clothing-jeans', 'mens-clothing-trousers', 'mens-clothing-coats', 'mens-clothing-jackets']
        elif domain == 1:
            dir_lst = ['womens-clothing-blouses-tunics', 'womens-clothing-pullovers-and-cardigans', 'womens-clothing-shirts', #'womens-clothing-dresses',
                                    'womens-clothing-jeans', 'womens-clothing-trousers', 'womens-clothing-jackets', 'womens-clothing-jackets-coats']
        elif domain == 2:
            dir_lst = ['men']
        elif domain == 3:
            dir_lst = ['women']
        elif domain == 4:
            dir_lst = ['all']
        else:
            raise ValueError('no suche domain')

        self.image_dir_lists = []
        self.length = 0

        for dir_name in dir_lst:
            glob_lst = glob.glob(os.path.join(root_path, dir_name, '*'))
            self.image_dir_lists += glob_lst
            self.length += len(glob_lst)
        if domain == 4:
            self.image_dir_lists = sorted(self.image_dir_lists, reverse=True)

        print(f'{self.name} : {self.length}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):        
        # random flip
        # random_flip = random.randint(0, 1) # random face flip

        dir_path = self.image_dir_lists[index]
        out_data = []
        for output_type in self.output_type_lst:
            if output_type in ['pose']:
                if self.domain in [0, 1]:
                    pose_path = os.path.join(dir_path, str(self.resolution), 'pose.npy')
                    pose_data = np.load(pose_path)
                else:
                    pose_data = np.array([
                        [128, 40],
                        [128, 65],

                        [105, 65],
                        [102, 100],
                        [99, 135],

                        [151, 65],
                        [154, 100],
                        [157, 135],

                        [110, 130],
                        [110, 170],
                        [110, 220],

                        [146, 130],
                        [146, 170],
                        [146, 220],

                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                    ])
                pose_data = kp_to_map(self.resolution, pose_data)
                out_data.append(pose_data)
            else:
                # get data as pil image
                if self.domain in [0, 1, 2, 3, 4]:
                    file_path = os.path.join(dir_path, str(self.resolution), f'{output_type}.png')
                    file_data = Image.open(file_path)
                else:
                    raise ValueError('no such domain')
                
                # horizontal flip
                # if random_flip:
                #     file_data = transforms.functional.hflip(file_data)
                
                # pil image to tensor and normalize
                if output_type in ['person', 'face', 'body']:
                    file_data = self.transform_3ch(file_data)
                elif output_type in ['face_seg', 'body_seg', 'person_seg']:
                    file_data = self.transform_1ch(file_data)
                elif output_type in ['person_mat']:
                    file_data = self.transform_4ch(file_data)
                else:
                    raise ValueError(f'No such output type: {output_type}')
                out_data.append(file_data)

        return out_data

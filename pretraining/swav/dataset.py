# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import random
import torch
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset

class MultiCropDataset(Dataset):
    def __init__(
        self,
        data_path,
        transforms
    ):
        super(MultiCropDataset, self).__init__()
        self.data_path = data_path
        
        manifest_file = os.path.join(data_path, 'manifest.pkl')
        if os.path.isfile(manifest_file):
            with open(manifest_file, mode='rb') as f:
                self.fpaths = pickle.load(f)
        else:
            self.fpaths = glob(data_path + '**/*')
            with open(manifest_file, mode='wb') as f:
                pickle.dump(self.fpaths, f)
        
        print(f'Found {len(self.fpaths)} images in dataset.')
        self.tfs = transforms
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        #get the filepath to load
        f = self.fpaths[index]
        
        # process multiple transformed crops of the image
        image = Image.open(f)
        multi_crops = list(map(
            lambda tfs: tfs(image), self.tfs
        ))

        return multi_crops

class RandomGaussianBlur:
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
    
class GaussNoise:
    """Gaussian Noise to be applied to images that have been scaled to fit in the range 0-1"""
    def __init__(self, var_limit=(1e-5, 1e-4), p=0.5):
        self.var_limit = np.log(var_limit)
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            image = image + torch.from_numpy(noise)
            image = torch.clamp(image, 0, 1)
        
        return image

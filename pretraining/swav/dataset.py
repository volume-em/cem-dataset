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
        data_dir,
        transforms,
        weight_gamma=None
    ):
        super(MultiCropDataset, self).__init__()
        self.data_dir = data_dir
        
        self.subdirs = []
        for sd in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, sd)):
                self.subdirs.append(sd)
        
        # images and masks as dicts ordered by subdirectory
        self.paths_dict = {}
        for sd in self.subdirs:
            sd_fps = glob(os.path.join(data_dir, f'{sd}/*.tiff'))
            if len(sd_fps) > 0:
                self.paths_dict[sd] = sd_fps
        
        # calculate weights per example, if weight gamma is not None
        self.weight_gamma = weight_gamma
        if weight_gamma is not None:
            self.weights = self._example_weights(self.paths_dict, gamma=weight_gamma)
        else:
            self.weights = None
        
        # unpack dicts to lists of images
        self.paths = []
        for paths in self.paths_dict.values():
            self.paths.extend(paths)
            
        print(f'Found {len(self.subdirs)} subdirectories with {len(self.paths)} images.')
        
        self.tfs = transforms
        
    def __len__(self):
        return len(self.paths)
    
    @staticmethod
    def _example_weights(paths_dict, gamma=0.3):
        # counts by source subdirectory
        counts = np.array(
            [len(paths) for paths in paths_dict.values()]
        )
        
        # invert and gamma the distribution
        weights = 1 / counts
        weights = weights ** gamma
        
        # for interpretation, normalize weights 
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights
        
        # repeat weights per n images
        example_weights = []
        for w,c in zip(weights, counts):
            example_weights.extend([w] * c)
            
        return torch.tensor(example_weights)

    def __getitem__(self, index):
        # get the filepath to load
        f = self.paths[index]
        
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

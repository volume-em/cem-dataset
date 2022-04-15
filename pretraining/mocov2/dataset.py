import os
import random
import torch
import numpy as np
import dask.array as da
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter
    
class EMData(Dataset):
    def __init__(
        self,
        data_dir,
        transforms,
        weight_gamma=None
    ):
        super(EMData, self).__init__()
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
    
    def __getitem__(self, idx):
        #get the filepath to load
        f = self.fpaths[idx]#.compute()
        
        #load the image and add an empty channel dim
        image = Image.open(f)
            
        #transform the images
        image1 = self.tfs(image)
        image2 = self.tfs(image)
        
        #return the two images as 1 tensor concatenated on
        #the channel dimension, we'll split it later
        return torch.cat([image1, image2], dim=0)

    def __getitem__(self, index):
        # get the filepath to load
        f = self.paths[index]
        
        # process multiple transformed crops of the image
        image = Image.open(f)
        
        # transform the images
        image1 = self.tfs(image)
        image2 = self.tfs(image)

        return torch.cat([image1, image2], dim=0)

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
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
import torch
import numpy as np
import dask.array as da
from skimage.io import imread
from glob import glob
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFilter
import random

class EMData(Dataset):
    """
    Dataset class for loading and augmenting unsupervised data.
    """
    
    def __init__(self, dirpath, tfs, fpaths_file=None, fixed_seed=None):
        super(EMData, self).__init__()
        self.dirpath = dirpath
        self.tfs = tfs
        self.fpaths_file = fpaths_file
        self.is_dask = False
        self.fixed_seed = fixed_seed
        
        #get all the tiff filepaths
        if fpaths_file is None:
            self.fpaths = glob(self.dirpath + '*.tiff')
            print(f'Found {len(self.fpaths)} tiff images in directory')
        else:
            if fpaths_file[-4:] == '.npy':
                self.fpaths = np.load(fpaths_file)
            elif fpaths_file[-4:] == '.npz':
                self.fpaths = da.from_npy_stack(fpaths_file)
                self.is_dask = True
                
            print(f'Loaded {fpaths_file} with {len(self.fpaths)} tiff images')
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, idx):
        #get the filepath to load
        if self.is_dask:
            f = self.fpaths[idx].compute()
        else:
            f = self.fpaths[idx]
        
        #load the image and add an empty channel dim
        #image = imread(f, 0).astype(np.uint8)[..., None]
        image = Image.open(f)
        
        #apply separate random augmentations to copies
        #of the image
        #image1 = self.tfs(image=image)['image']
        #image2 = self.tfs(image=image)['image']
        #fix seed
        random.seed(self.fixed_seed)
        image1 = self.tfs(image)
        image2 = self.tfs(image)
        
        #return the two images as 1 tensor concatenated on
        #the channel dimension, we'll split it later
        return torch.cat([image1, image2], dim=0)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class GaussNoise:
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

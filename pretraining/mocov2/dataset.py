import random
import torch
import numpy as np
import dask.array as da
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter

class EMData(Dataset):
    """
    Dataset class for loading and augmenting unsupervised data.
    """
    
    def __init__(self, fpaths_dask_array, tfs):
        super(EMData, self).__init__()
        self.fpaths_dask_array = fpaths_dask_array
        self.tfs = tfs
        
        self.fpaths = da.from_npy_stack(fpaths_dask_array)
        print(f'Loaded {fpaths_dask_array} with {len(self.fpaths)} tiff images')
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, idx):
        #get the filepath to load
        f = self.fpaths[idx].compute()
        
        #load the image and add an empty channel dim
        image = Image.open(f)
            
        #transform the images
        image1 = self.tfs(image)
        image2 = self.tfs(image)
        
        #return the two images as 1 tensor concatenated on
        #the channel dimension, we'll split it later
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
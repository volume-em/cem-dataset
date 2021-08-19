import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from albumentations import ImageOnlyTransform

class ContrastData(Dataset):
    def __init__(
        self,
        imdir,
        space_tfs,
        view1_color_tfs,
        view2_color_tfs=None
    ):
        super(ContrastData, self).__init__()

        self.imdir = imdir
        self.fpaths = glob(os.path.join(imdir, '*.tiff'), recursive=True)
        #self.fnames = os.listdir(imdir)
        
        print(f'Found {len(self.fpaths)} images in directory')

        #crops, resizes, flips, rotations, etc.
        self.space_tfs = space_tfs

        #brightness, contrast, jitter, blur, and
        #normalization
        self.view1_color_tfs = view1_color_tfs
        self.view2_color_tfs = view2_color_tfs

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        #fpath = os.path.join(self.imdir, self.fnames[idx])
        image = cv2.imread(fpath, 0)

        y = np.arange(0, image.shape[0], dtype=np.float32)
        x = np.arange(0, image.shape[1], dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x)
        grid_y, grid_x = grid_y.T, grid_x.T

        #space transforms treat coordinate grid like an image
        #bilinear interp is good, nearest would be bad
        view1_data = self.space_tfs(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])
        view2_data = self.space_tfs(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])

        view1 = view1_data['image']
        view1_grid = np.concatenate([view1_data['grid_y'], view1_data['grid_x']], axis=-1)
        view2 = view2_data['image']
        view2_grid = np.concatenate([view2_data['grid_y'], view2_data['grid_x']], axis=-1)

        view1 = self.view1_color_tfs(image=view1)['image']
        if self.view2_color_tfs is not None:
            view2 = self.view2_color_tfs(image=view2)['image']
        else:
            view2 = self.view1_color_tfs(image=view2)['image']

        output = {
            'fpath': fpath,
            'view1': view1,
            'view1_grid': torch.from_numpy(view1_grid).permute(2, 0, 1),
            'view2': view2,
            'view2_grid': torch.from_numpy(view2_grid).permute(2, 0, 1)
        }

        return output

class Grayscale(ImageOnlyTransform):
    """
    Resizes an image, but not the mask, to be divisible by a specific
    number like 32. Necessary for evaluation with segmentation models
    that use downsampling.
    """
    def __init__(self, channels=1, always_apply=True, p=1.0):
        super(Grayscale, self).__init__(always_apply, p)
        self.channels = channels

    def apply(self, img, **params):
        if img.ndim == 2:
            img = np.repeat(img[..., None], self.channels, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., 0]
        return img

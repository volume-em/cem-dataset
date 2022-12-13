import os, sys, math
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def convert_to_byte(image):
    """
    Verify that image is byte type
    """
    if image.dtype == np.uint8:
        return image
    else:
        image = image.astype(np.float32)
        image -= image.min()
        im_max = image.max()
        if im_max > 0: # avoid zero division
            image /= im_max
            
        image *= 255
        
        return image.astype(np.uint8)
    
def sparse_roi_boxes(
    reference_volume,
    roi_size,
    padding_value=0
    min_frac=0.7
):
    """
    Finds all ROIs of a given size within an overview volume
    that contain at least some non-padding values.
    
    Arguments:
    ----------
    reference_volume (np.ndarray): A low-resolution overview image
    that can fit in memory. Typically this is the lowest-resolution
    available in an image pyramid.
    
    roi_size (Tuple[d, h, w]): Size of ROIs in voxels relative to 
    the reference volume.
    
    padding_value (float): Value used to pad the reference volume.
    Must be confirmed by manual inspection.
    
    min_frac (float): Fraction from 0-1 of and ROI that must be
    non-padding values.
    
    Returns:
    --------
    roi_boxes (np.ndarray): Array of (N, 6) defining bounding boxes
    for ROIs that passed that min_frac condition.
    
    """
    # grid for box indices
    xcs, ycs, zcs = roi_size
    xsize, ysize, zsize = reference_volume.shape
    
    xgrid = np.arange(0, xsize + 1, xcs)
    ygrid = np.arange(0, ysize + 1, ycs)
    zgrid = np.arange(0, zsize + 1, zcs)
    
    max_padding = (1 - min_frac) * np.prod(roi_size)

    # make sure that there's always an ending index
    # so we have complete ranges
    if len(xgrid) < 2 or xsize % xcs > 0.5 * xcs:
        xgrid = np.append(xgrid, np.array(xsize)[None], axis=0)
    if len(ygrid) < 2 or ysize % ycs > 0.5 * ycs:
        ygrid = np.append(ygrid, np.array(ysize)[None], axis=0)
    if len(zgrid) < 2 or zsize % ycs > 0.5 * zcs:
        zgrid = np.append(zgrid, np.array(zsize)[None], axis=0)

    roi_boxes = []
    for xi, xf in zip(xgrid[:-1], xgrid[1:]):
        for yi, yf in zip(ygrid[:-1], ygrid[1:]):
            for zi, zf in zip(zgrid[:-1], zgrid[1:]):
                box_slices = tuple([slice(xi, xf), slice(yi, yf), slice(zi, zf)])
                n_not_padding = np.count_nonzero(
                    reference_volume[box_slices] == padding_value
                )
                if n_not_padding < max_padding:
                    roi_boxes.append([xi, yi, zi, xf, yf, zf])
                    
    return np.array(roi_boxes)

def crop_volume(
    volume, 
    volume_name, 
    resolution, 
    save_path,
    cube_size=256,
    padding_value=0,
    min_frac=0.7
):
    # find possible ROIs that are not just blank padding 
    # they may still be uniform resin though
    roi_boxes = sparse_roi_boxes(
        volume, cube_size, padding_value, min_frac
    )
    
    # randomly select n_cubes ROI boxes
    box_indices = np.random.choice(
        range(len(roi_boxes)), size=(min(n_cubes, len(roi_boxes)),), replace=False
    )
    roi_boxes = roi_boxes[box_indices]

    # loop through the boxes that we selected
    for bbox in tqdm(roi_boxes):
        x1, y1, z1, x2, y2, z2 = bbox
        bbox_slices = [
            slice(x1, x2),
            slice(y1, y2),
            slice(z1, z2)
        ]
        
        # crop the cube
        cube = volume[bbox_slices]
        cube = convert_to_byte(cube)

        cube_fname = f'{volume_name}-ROI-x{x1}-{x2}_y{y1}-{y2}_z{z1}-{z2}.nrrd'
        cube_fpath = os.path.join(save_path, cube_fname)
        
        cube = sitk.GetImageFromArray(cube)
        cube.SetSpacing(resolution)
        sitk.WriteImage(cube, cube_fpath)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str, help='path to a volume file (nrrd or mrc)')
    parser.add_argument('save_path', type=str, help='path to save the volumes')
    parser.add_argument('-cs', type=int, default=256, help='dimension of cubes to crop')
    parser.add_argument('-gb', type=float, default=5, help='maximum number of GBs to crop')
    args = parser.parse_args()
    
    directory = args.directory
    save_path = args.save_path
    target_dir = args.target_dir
    cube_size = args.cs
    max_gbs = args.max_gbs
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    # load the image into array
    volume = sitk.ReadImage(fp)
    resolution = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume)
    
    # crop into cubes
    volname = '.'.join(os.path.basename(fp).split('.')[:-1])
    crop_volume(
        volume,
        volname,
        resolution,
        save_path,
        cube_size,
        padding_value=0
    )
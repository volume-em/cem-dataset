"""
Description:
------------

This script accepts a directory with image volume files and slices cross sections 
from the given axes (xy, xz, yz). The resultant cross sections are saved in 
the given save directory.

Importantly, the saved image files are given a slightly different filename:
We add '-LOC-{axis}_{slice_index}' to the end of the filename, where axis denotes the
cross-sectioning plane (0->xy, 1->xz, 2->yz) and the slice index is the position of
the cross-section on that axis. Once images from 2d and 3d datasets
start getting mixed together, it can be difficult to keep track of the
provenance of each patch. Everything that appears before '-LOC-' is the
name of the original dataset, the axis and slice index allow us to lookup the
exact location of the cross-section in the volume.

Example usage:
--------------

python cross_section3d.py {imdir} {savedir} --axes 0 1 2 --spacing 1 --processes 4

For help with arguments:
------------------------

python cross_section3d.py --help
"""

import os
import math
import pickle
import argparse
import numpy as np
from glob import glob
from skimage import io
from multiprocessing import Pool

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("int16"): 32767,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

def patch_crop(image, crop_size=224):
    if image.ndim == 3:
        if image.shape[2] not in [1, 3]:
            print('Accidentally 3d?', image.shape)
        image = image[..., 0]
        
    # at least 1 image patch
    ysize, xsize = image.shape
    ny = max(1, int(round(ysize / crop_size)))
    nx = max(1, int(round(xsize / crop_size)))
    
    patches = []
    locs = []
    for y in range(ny):
        # start and end indices for y
        ys = y * crop_size
        ye = min(ys + crop_size, ysize)
        for x in range(nx):
            # start and end indices for x
            xs = x * crop_size
            xe = min(xs + crop_size, xsize)
            
            # crop the patch
            patch = image[ys:ye, xs:xe]

            patches.append(patch)
            locs.append(f'{ys}-{ye}_{xs}-{xe}')

    return patches, locs

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing 2d image files')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save the patch files')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=224,
                        help='Size of square image patches. Default 224.')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    # read in the parser arguments
    imdir = args.imdir
    savedir = args.savedir
    crop_size = args.crop_size
    processes = args.processes
    
    # check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    # get the list of all images (png, jpg, tif)
    fpath_groups = {}
    for sd in glob(os.path.join(imdir, '*')):
        if os.path.isdir(sd):
            sd_name = os.path.basename(sd)
            fpath_groups[sd_name] = [fp for fp in glob(os.path.join(sd, '*')) if not os.path.isdir(fp)]

    print(f'Found {len(fpath_groups.keys())} image groups to process')

    subdirs = list(fpath_groups.keys())
    fpath_lists = list(fpath_groups.values())

    def create_patches(*args):
        subdir, fpaths = args[0]

        exp_name = subdir
        patch_dict = {'names': [], 'patches': []}
        zpad = 1 + math.ceil(math.log(len(fpaths)))

        # check if results have already been generated,
        # skip this image if so. useful for resuming
        out_path = os.path.join(savedir, exp_name + '.pkl')
        if os.path.isfile(out_path):
            print(f'Already processed {fp}, skipping!')
            return

        for ix,fp in enumerate(fpaths):
            # try to load the image, if it's not possible
            # then pass but print
            try:
                im = io.imread(fp)
            except:
                print('Failed to open: ', fp)
                return
            
            assert (im.min() >= 0), 'Negative images not allowed!'
            
            if im.dtype != np.uint8:
                dtype = im.dtype
                max_value = MAX_VALUES_BY_DTYPE[dtype]
                im = im.astype(np.float32) / max_value
                im = (im * 255).astype(np.uint8)
                        
            # crop the image into patches
            patches, locs = patch_crop(im, crop_size)

            # appropriate filenames with location
            imname = str(ix).zfill(zpad)
            names = []
            for loc_str in locs:
                # add the -LOC- to indicate the point of separation between
                # the dataset name and the slice location information
                patch_loc_str = f'-LOC-2d-{loc_str}'
                names.append(imname + patch_loc_str)
                
            # store results in patch_dict
            patch_dict['names'].extend(names)
            patch_dict['patches'].extend(patches)
                    
        with open(out_path, 'wb') as handle:
            pickle.dump(patch_dict, handle)
    
    with Pool(processes) as pool:
        pool.map(create_patches, zip(subdirs, fpath_lists))

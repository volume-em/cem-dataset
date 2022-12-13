"""
Description:
------------

This script accepts a directory with 2d images and processes them
to make sure they all have unsigned 8-bit pixels and are grayscale. 

The resultant images 
are saved in the given save directory.

Importantly, the saved image files are given a slightly different filename:
We add '-LOC-2d' to the end of the filename. Once images from 2d and 3d datasets
start getting mixed together, it can be difficult to keep track of the
provenance of each patch. Everything that appears before '-LOC-2d' is the
name of the original dataset and we, of course, know that that dataset is
a 2d EM image.

Example usage:
--------------

python cleanup2d.py {imdir} -o {savedir} -p 4

For help with arguments:
------------------------

python cleanup2d.py --help
"""

import os, math
import argparse
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from multiprocessing import Pool

if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing subdirectories of 2d EM images')
    parser.add_argument('-o', type=str, metavar='savedir', dest='savedir', 
                        help='Path to save the processed images as copies, if not given images will be overwritten.')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    imdir = args.imdir
    processes = args.processes
    
    savedir = args.savedir
    if savedir is None:
        savedir = args.imdir
    else:
        os.makedirs(savedir, exist_ok=True)
    
    # get the list of all images (png, jpg, tif, etc.)
    fpath_groups = {}
    for sd in glob(os.path.join(imdir, '*')):
        if os.path.isdir(sd):
            sd_name = os.path.basename(sd)
            fpath_groups[sd_name] = [fp for fp in glob(os.path.join(sd, '*')) if not os.path.isdir(fp)]

    print(f'Found {len(fpath_groups.keys())} image groups to process')

    subdirs = list(fpath_groups.keys())
    fpath_lists = list(fpath_groups.values())
    
    def process_image(*args):
        subdir, fpaths = args[0]
        for fp in fpaths:
            try:
                im = imread(fp)

                # has to be 2d image with or without channels
                assert im.ndim < 4

                # make sure last dim is channel
                if im.ndim == 3:
                    assert im.shape[-1] <= 4
                    im = im[..., 0]

            except:
                print('Failed to read: ', fp)
                pass

            dtype = str(im.dtype)

            is_float = False
            unsigned = False
            if dtype[0] == 'u':
                unsigned = True
            elif dtype[0] == 'f':
                is_float = True

            if dtype == 'uint8': # nothing to do
                pass
            else:
                # get the bitdepth
                bits = int(dtype[-2]) # 16, 32, or 64

                # explicitly convert the image to float
                im = im.astype('float') 

                if unsigned: 
                    # unsigned conversion just requires division by max and
                    # multiplication by 255
                    im /= (2 ** bits)
                    im *= 255
                elif unsigned is False and is_float is False:
                    # signed conversion adds the additional step of
                    # subtracting the minimum negative value
                    im -= -(2 ** (bits - 1))
                    im /= (2 ** bits)
                    im *= 255
                else:
                    # we're working with float.
                    # because the range is variable, we'll just subtract
                    # the minimum, divide by maximum, and multiply by 255
                    im -= im.min()
                    im /= im.max()
                    im *= 255

                im = im.astype(np.uint8)

            # save the processed image to the new directory
            outdir = os.path.join(savedir, subdir)
            if not os.path.exists(outdir):
                os.mkdir(outdir)
                
            imsave(os.path.join(outdir, im_name), im, check_contrast=False)
    
    with Pool(processes) as pool:
        pool.map(process_image, zip(subdirs, fpath_lists))
        
    print('Finished 2D image cleanup.')
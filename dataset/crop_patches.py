"""
Description:
------------

It is assumed that this script will be run after the cross_sections.py script. Errors
may arise if this is not the case.

This script takes a directory containing 2d tiff images and crops those large images
into squares of a given dimension (default 224). In addition to creating the patch,
it also calculates and saves a difference hash for that patch. Doing both in a single
step significantly cuts down on I/O time. Both the patch and the hash are saved
in the same directory. All patches are .tiff and all hashes are .npy

Example usage:
--------------

python crop_patches.py {imdir} {patchdir} --crop_size 224 --hash_size 8 --processes 4

For help with arguments:
------------------------

python crop_patches.py --help
"""

import os
import argparse
import imagehash
import numpy as np
from skimage.io import imread, imsave
from glob import glob
from PIL import Image
from multiprocessing import Pool
    
def calculate_hash(image, crop_size, hash_size=8):
    #Creates the dhash for the resized image
    #this guarantees that smaller images are not more likely
    #to be recognized as unique
    return imagehash.dhash(Image.fromarray(image).resize(crop_size, crop_size), hash_size=hash_size).hash

def patch_and_hash(impath, patchdir, crop_size=224, hash_size=8):
    #load the image
    image = imread(impath)
    
    #handle rgb by keeping only the first channel
    if image.ndim == 3:
        image = image[..., 0]
        
    #assumes that we are working with the output of cross_sections.py
    #which saves all images as .tiff
    prefix = impath.split('/')[-1].split('.tiff')[0]
    
    #get the image size
    ysize, xsize = image.shape
    
    #this means that the smallest allowed image patch must have
    #at least half of the desired crop size in both dimensions
    ny = max(1, int(round(ysize / crop_size)))
    nx = max(1, int(round(xsize / crop_size)))
    
    for y in range(ny):
        #start and end indices for y
        ys = y * crop_size
        ye = min(ys + crop_size, ysize)
        for x in range(nx):
            #start and end indices for x
            xs = x * crop_size
            xe = min(xs + crop_size, xsize)
            
            #crop the patch and calculate its hash
            patch = image[ys:ye, xs:xe]
            patch_hash = calculate_hash(patch, crop_size, hash_size)
                
            #make the output file paths
            patch_path = os.path.join(patchdir, f'{prefix}_{ys}_{xs}.tiff')
            hash_path = patch_path.replace('.tiff', '.npy')
            
            #save the patch and the hash
            imsave(patch_path, patch, check_contrast=False)
            np.save(hash_path, patch_hash)

#main function of the script
if __name__ == "__main__":
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing tiff images')
    parser.add_argument('patchdir', type=str, metavar='patchdir', help='Directory in which to save cropped patches')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=224,
                        help='Size of square image patches. Default 224.')
    parser.add_argument('-hs', '--hash_size', dest='hash_size', type=int, metavar='hash_size', default=8,
                        help='Size of the image hash. Default 8.')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=32,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    #read in the parser arguments
    imdir = args.imdir
    patchdir = args.patchdir
    crop_size = args.crop_size
    hash_size = args.hash_size
    processes = args.processes
    
    #make sure the patchdir exists
    if not os.path.isdir(patchdir):
        os.mkdir(patchdir)
    
    #get the list of all tiff files in the imdir
    impaths = np.sort(glob(os.path.join(imdir, '*.tiff')))
    print(f'Found {len(impaths)} tiff images to crop.')
    
    def map_func(impath):
        patch_and_hash(impath, patchdir, crop_size, hash_size)    
        return None
    
    #loop over the images and save patches and hashes
    #using the given number of processes
    with Pool(processes) as pool:
        pool.map(map_func, impaths)
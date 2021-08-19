"""
Description:
------------

This script accepts a directory with 2d images and processes them
to make sure they all have unsigned 8-bit pixels. The resultant images 
are saved in the given save directory.

Importantly, the saved image files are given a slightly different filename:
We add '-LOC-2d' to the end of the filename. Once images from 2d and 3d datasets
start getting mixed together, it can be difficult to keep track of the
provenance of each patch. Everything that appears before '-LOC-2d' is the
name of the original dataset and we, of course, know that that dataset is
a 2d EM image.

Example usage:
--------------

python cleanup2d.py {imdir} {savedir} --processes 4

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

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing 2d EM images')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save the processed images')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    #read in the parser arguments
    imdir = args.imdir
    savedir = args.savedir
    processes = args.processes
    
    #check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    #get the list of all images of any format
    fpaths = np.array(glob(imdir + '*'))
    
    print(f'Found {len(fpaths)} images')
    
    #loop over each fpath and save the slices
    def process_image(fp):
        #try to read the image, if it's not possible then pass
        try:
            im = imread(fp)
        except:
            print('Failed to open: ', fp)
            pass
        
        #first check if we're working with signed or unsigned pixels
        dtype = str(im.dtype)
        
        is_float = False
        unsigned = False
        if dtype[0] == 'u':
            unsigned = True
        elif dtype[0] == 'f':
            is_float = True
            
        if dtype == 'uint8': #nothing to do
            pass
        else:
            #get the number of bits per pixel
            bits = int(dtype[-2]) #16, 32, or 64
            
            #explicitly convert the image to float
            im = im.astype('float') 
            
            if unsigned: 
                #unsigned conversion just requires division by max and
                #multiplication by 255
                im /= (2 ** bits) #scales from 0-1
                im *= 255
            elif unsigned is False and is_float is False:
                #signed conversion adds the additional step of
                #subtracting the minimum negative value
                im -= -(2 ** (bits - 1))
                im /= (2 ** bits) #scales from 0-1
                im *= 255
            else:
                #this means we're working with float.
                #because the range is variable, we'll just subtract
                #the minimum, divide by maximum, and multiply by 255
                #this means the pixel range is always 0-255
                im -= im.min()
                im /= im.max()
                im *= 255

            #convert to uint8
            im = im.astype(np.uint8)
        
        #establish a filename prefix from the filepath
        fext = fp.split('.')[-1]
        exp_name = fp.split('/')[-1].split(f'.{fext}')[0]
        
        #create a new filename with the -LOC- identifier to help
        #find the experiment name in later scripts, the 2d helps
        #to distinguish this image from the cross sections of 3d volumes
        im_name = f'{exp_name}-LOC-2d.tiff'
        
        #save the processed image
        imsave(os.path.join(savedir, im_name), im, check_contrast=False)
    
    #running the function with multiple processes
    #results in a much faster runtime
    with Pool(processes) as pool:
        pool.map(process_image, fpaths)
        
    print('Finished')
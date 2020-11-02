"""
Description:
------------

This script accepts a directory with image volume files
and slices cross sections from the given axes (xy, xz, yz).
The resultant cross sections are saved in the given directory

Example usage:
--------------

python cross_section3d.py {imdir} {savedir} --axes 0 1 2 --spacing 1 --processes 4

For help with arguments:
------------------------

python cross_section3d.py --help
"""

import os, math
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from skimage.io import imsave
from multiprocessing import Pool

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("int16"): 32767,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing volume files')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save the cross sections')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-xy, 1-xz, 2-yz)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    #read in the parser arguments
    imdir = args.imdir
    savedir = args.savedir
    axes = args.axes
    spacing = args.spacing
    processes = args.processes
    
    #check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    #get the list of all volumes (mrc, tif, nrrd, nii.gz)
    fpaths = np.array(glob(imdir + '*'))
    
    print(f'Found {len(fpaths)} image volumes')
    
    #loop over each fpath and save the slices
    def create_slices(fp):
        #try to load the volume, if it's not possible
        #then pass
        try:
            im = sitk.ReadImage(fp)
            print(im.GetSize(), fp)
        except:
            print('Failed to open: ', fp)
            pass
        
        #extract the pixel size from the volume
        #if the z-pixel size is more than 20% different
        #from the x-pixel size, don't slice over orthogonal
        #directions
        pixel_sizes = im.GetSpacing()
        anisotropy = np.abs(pixel_sizes[0] - pixel_sizes[2]) / pixel_sizes[0]
        
        #convert the volume to numpy 
        im = sitk.GetArrayFromImage(im)
        
        assert (im.min() >= 0), 'Negative images not allowed!'
        
        #check if the volume is uint8, convert if not
        if im.dtype != np.uint8:
            dtype = im.dtype
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            im = im.astype(np.float32) / max_value
            im = (im * 255).astype(np.uint8)
        
        #establish a filename prefix from the imvolume
        #extract the experiment name from the filepath
        fext = fp.split('.')[-1]
        exp_name = fp.split('/')[-1].split(f'.{fext}')[0]
        
        #loop over the axes and save slices
        for axis in axes:
            #only process xy slices if the volume is anisotropic
            if anisotropy > 0.2 and axis != 0:
                continue
                
            #get the axis dimension and get evenly spaced slice indices
            nmax = im.shape[axis] - 1
            slice_indices = np.arange(0, nmax, spacing, dtype=np.long)
            
            #for the index naming convention we want to pad all slice indices
            #with zeros up to some length: eg. 1 --> 0001 to match 999 --> 0999
            #we get the number of zeros from nmax
            zpad = math.ceil(math.log(nmax, 10))
            
            for idx in slice_indices:
                index_str = str(idx).zfill(zpad)
                #add the -LOC- to indicate the point of separation between
                #the dataset name and the slice location information
                slice_name = f'{exp_name}-LOC-{axis}_{index_str}.tiff'
                
                #don't save anything if the slice already exists
                if os.path.isfile(os.path.join(savedir, slice_name)):       
                    continue
                
                #slice the volume on the proper axis
                if axis == 0:
                    im_slice = im[idx]
                elif axis == 1:
                    im_slice = im[:, idx]
                else:
                    im_slice = im[:, :, idx]
                    
                #save the slice
                imsave(os.path.join(savedir, slice_name), im_slice, check_contrast=False)
    
    #running the function with multiple processes
    #results in a much faster runtime
    with Pool(processes) as pool:
        pool.map(create_slices, fpaths)
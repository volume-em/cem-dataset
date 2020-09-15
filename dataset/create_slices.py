import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
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
    parser.add_argument('impath', type=str, metavar='impath', help='Directory containing volume files')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the slices')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0],
                        help='Volume axes along which to slice (0-xy, 1-xz, 2-yz)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    parser.add_argument('-n', '--span', dest='span', type=int, metavar='span', default=1,
                        help='Span of contiguous slices to cut at every spacing interval')
    

    args = parser.parse_args()

    #read in the parser arguments
    impath = args.impath
    save_path = args.save_path
    axes = args.axes
    spacing = args.spacing
    span = args.span
    
    #check if the save path exists, if not
    #create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #get the list of all volumes (mrc, tif, nrrd, hdf)
    fpaths = np.array(glob(impath + '*.mrc'))
    fpaths = np.concatenate([fpaths, np.array(glob(impath + '*.tif'))])
    fpaths = np.concatenate([fpaths, np.array(glob(impath + '*.nrrd'))])
    fpaths = np.concatenate([fpaths, np.array(glob(impath + '*.hdf'))])
    
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
            #get the axis dimension and get evenly spaced slice indices
            nmax = im.shape[axis] - 1
            
            slice_indices = np.concatenate(
                [np.arange(max(c - span//2, 0), min(c + span//2+1, nmax)) for c in np.arange(0, nmax, spacing)]
            )
            
            for idx in slice_indices:
                idx = int(idx)
                slice_name = '_'.join([exp_name, str(axis), str(idx).zfill(4) + '.tiff'])
                #don't save anything if the slice already exists
                if os.path.exists(save_path + slice_name):       
                    continue
                
                if axis == 0:
                    im_slice = im[idx]
                elif axis == 1:
                    im_slice = im[:, idx]
                else:
                    im_slice = im[:, :, idx]
                    
                imsave(save_path + slice_name, im_slice)
                        
    with Pool(4) as pool:
        pool.map(create_slices, fpaths)
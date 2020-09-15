import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('impath', type=str, metavar='impath', help='Directory containing volume files')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the slices')

    args = parser.parse_args()

    #read in the parser arguments
    impath = args.impath
    save_path = args.save_path
    crop_size = 128
    min_size = crop_size // 2
    
    #check if the save path exists, if not
    #create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #get the list of all volumes (mrc, tif, nrrd, hdf)
    fpaths = np.array(glob(impath + '*.mrc'))
    fpaths = np.concatenate([fpaths, np.array(glob(impath + '*.tif'))])
    fpaths = np.concatenate([fpaths, np.array(glob(impath + '*.nrrd'))])
    
    print(f'Found {len(fpaths)} image volumes')
    
    #loop over each fpath and save the slices
    for fp in fpaths:
        #read the image
        try:
            im = sitk.ReadImage(fp)
            im = sitk.GetArrayFromImage(im)
            im = im[..., 0] if im.ndim == 4 else im
            print(im.shape, fp)
        except:
            print('Failed to open: ', fp)
            pass

        #establish a filename prefix from the imvolume
        #extract the experiment name from the filepath
        exp_name = '.'.join(fp.split('/')[-1].split('.')[:-1])
        
        #loop over the axes and save slices
        xmax, ymax, zmax = im.shape
        for x in np.arange(0, xmax, crop_size):
            for y in np.arange(0, ymax, crop_size):
                for z in np.arange(0, zmax, crop_size):
                    dest_path = save_path + exp_name + f'_{z}_{y}_{x}.npy'
                    if os.path.isfile(dest_path):
                        continue
                    elif x < xmax - min_size or y < ymax - min_size  or z < zmax - min_size:
                        xe = min(x + crop_size, xmax)
                        ye = min(y + crop_size, ymax)
                        ze = min(z + crop_size, zmax)
                        np.save(save_path + exp_name + f'_{z}_{y}_{x}.npy', im[x:xe, y:ye, z:ze])

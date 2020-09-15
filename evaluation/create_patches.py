import os
import argparse
import numpy as np
from skimage.io import imread, imsave
from glob import glob

    
def patch_crop(impath, dst_dir, crop_size, label):
    #load the images
    image = imread(impath)
        
    prefix = '.'.join(impath.split('/')[-1].split('.')[:-1])
    fext = impath.split('/')[-1].split('.')[-1]
    
    #get the image size
    ysize, xsize = image.shape
    
    #get the y starts and ends
    if ysize % crop_size == 0:
        ystarts = np.arange(0, ysize, crop_size)
        yends = ystarts + crop_size
    else:
        ny = (ysize // crop_size) + 1
        overlap = ny * crop_size - ysize
        ystarts = np.arange(0, ysize, crop_size) - np.linspace(0, overlap, ny, dtype=np.uint8)
        yends = ystarts + crop_size
    
    #get the x start and ends
    if xsize % crop_size == 0:
        xstarts = np.arange(0, xsize, crop_size)
        xends = xstarts + crop_size
    else:
        nx = (xsize // crop_size) + 1
        overlap = nx * crop_size - xsize
        xstarts = np.arange(0, xsize, crop_size) - np.linspace(0, overlap, nx, dtype=np.uint8)
        xends = xstarts + crop_size
    
    for ys, ye in zip(ystarts, yends):
        for xs, xe in zip(xstarts, xends):
            patch = image[ys:ye, xs:xe]
            
            #don't save black images
            if patch.max() == 0:
                continue
            
            if label != 0:
                patch = (patch == label).astype(np.uint8)
                
            fname = os.path.join(dst_dir, f'{prefix}_{ys}_{xs}.{fext}')
            imsave(fname, patch, check_contrast=False)

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Directory containing 2dimage files')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing images')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the slices')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=256,
                        help='Size of square image patches')
    parser.add_argument('-l', '--mask_label', dest='mask_label', type=int, metavar='mask_label', default=0,
                        help='Mask label to save')
    

    args = parser.parse_args()

    #read in the parser arguments
    imdir = args.imdir
    save_path = args.save_path
    crop_size = args.crop_size
    label = args.mask_label
    
    #check if the save path exists, if not
    #create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #get the list of all tiff files
    impaths = glob(os.path.join(imdir, '*'))
    print(f'Found {len(impaths)} images to crop.')
    
    #loop over the images and save the images that pass a filter
    for ix, impath in enumerate(impaths):
        patch_crop(impath, save_path, crop_size, label)
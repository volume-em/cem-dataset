import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('impath', type=str, metavar='impath', help='Path to an image file')
    parser.add_argument('mskpath', type=str, metavar='mskpath', help='Path to labelmap file')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the slices')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[2],
                        help='Volume axes along which to slice (0-yz, 1-xz, 2-xy)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')

    args = vars(parser.parse_args())
    
    return args


if __name__ == '__main__':
    args = parse_args()

    #read in the parser arguments
    impath = args['impath']
    mskpath = args['mskpath']
    save_path = args['save_path']
    axes = args['axes']
    spacing = args['spacing']

    #create images and masks directories in save_path
    #if they do not already exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(os.path.join(save_path, 'images')):
        os.mkdir(os.path.join(save_path, 'images'))

    if not os.path.exists(os.path.join(save_path, 'masks')):
        os.mkdir(os.path.join(save_path, 'masks'))
    
    #load the image and labelmap volumes
    image = sitk.ReadImage(impath)
    labelmap = sitk.ReadImage(mskpath)
    
    #establish a filename prefix from the impath
    exp_name = impath.split('/')[-1].split('.')[0]

    #loop over the axes and save slices
    for axis in axes:
        #get the axis dimension and get evenly spaced slice indices
        slice_indices = np.arange(0, image.GetSize()[axis] - 1, spacing, dtype=np.long)
        for idx in slice_indices:
            idx = int(idx)
            slice_name = '_'.join([exp_name, str(axis), str(idx)])

            if axis == 0:
                image_slice = image[idx]
                if mskpath != 'none':
                    labelmap_slice = labelmap[idx]
            elif axis == 1:
                image_slice = image[:, idx]
                if mskpath != 'none':
                    labelmap_slice = labelmap[:, idx]
            else:
                image_slice = image[:, :, idx]
                if mskpath != 'none':
                    labelmap_slice = labelmap[:, :, idx]

            sitk.WriteImage(image_slice, os.path.join(save_path, f'images/{slice_name}.tiff'))
            sitk.WriteImage(labelmap_slice, os.path.join(save_path, f'masks/{slice_name}.tiff'))

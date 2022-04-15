"""
Description:
------------

This script accepts a directory with image volume files and slices cross sections 
from the given axes (xy, xz, yz). Then it patches the cross-sections into many smaller
images. All patches from a dataset a deduplicated such that patches with nearly identical
content are filtered out.

Patches along with filenames are stored in a dictionary and saved as pickle
files. Importantly, filenames follow the convention:

'{dataset_name}-LOC-{slicing_axis}_{slice_index}_{h1}-{h2}_{w1}-{w2}'

Slicing axis denotes the cross-sectioning plane (0->xy, 1->xz, 2->yz). Slice index
is the index of the image along the slicing axis. h1,h2 are start and end rows and 
w1,w2 are start and end columns. This gives enough information to precisely locate
the patch in the original 3D dataset.

Lastly, if the directory of 3D datasets includes a mixture of isotropic and anisotropic
volumes it is important that each dataset has a correct header recording the voxel
size. This script uses SimpleITK to read the header. If z resolution is more that 25% 
different than xy resolution, then cross-sections will only be cut from the xy plane
even if axes 0, 1, 2 are passed to the script (see usage example below). 

Likewise, if there are video files as well, it is essential that they have the word 'video' 
somewhere in the filename.

Example usage:
--------------

python patchify3d.py {imdir} {savedir} --axes 0 1 2 --spacing 1 --processes 4

For help with arguments:
------------------------

python patchify3d.py --help

"""

import os
import math
import pickle
import argparse
import imagehash
import numpy as np
import SimpleITK as sitk
from glob import glob
from PIL import Image
from skimage import io
from multiprocessing import Pool

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("int16"): 32767,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

def calculate_hash(image, crop_size, hash_size=8):
    # calculate the hash on the resized image
    imsize = (crop_size, crop_size)
    pil_image = Image.fromarray(image).resize(imsize, resample=2)
    
    return imagehash.dhash(pil_image, hash_size=hash_size).hash

def patch_and_hash(image, crop_size=224, hash_size=8):
    if image.ndim == 3:
        image = image[..., 0]
        
    # at least 1 image patch of any size
    ysize, xsize = image.shape
    ny = max(1, int(round(ysize / crop_size)))
    nx = max(1, int(round(xsize / crop_size)))
    
    patches = []
    hashes = []
    locs = []
    for y in range(ny):
        # start and end indices for y
        ys = y * crop_size
        ye = min(ys + crop_size, ysize)
        for x in range(nx):
            # start and end indices for x
            xs = x * crop_size
            xe = min(xs + crop_size, xsize)
            
            # crop the patch and calculate its hash
            patch = image[ys:ye, xs:xe]
            patch_hash = calculate_hash(patch, crop_size, hash_size)

            patches.append(patch)
            hashes.append(patch_hash)
            locs.append(f'{ys}-{ye}_{xs}-{xe}')

    return patches, hashes, locs

def deduplicate(patch_dict, min_distance):
    # all hashes are the same size
    hashes = np.array(patch_dict['hashes'])
    hashes = hashes.reshape(len(hashes), -1)

    # randomly permute the hashes such that we'll have random ordering
    random_indices = np.random.permutation(np.arange(0,  len(hashes)))
    hashes = hashes[random_indices]

    exemplars = []
    while len(hashes) > 0:
        ref_hash = hashes[0]

        # a match has Hamming distance less than min_distance
        matches = np.where(
            np.logical_xor(ref_hash, hashes).sum(1) <= min_distance
        )[0]

        # ref_hash is the exemplar (i.e. first in matches)
        exemplars.append(random_indices[matches[0]])

        # remove all the matched images from both hashes and indices
        hashes = np.delete(hashes, matches, axis=0)
        random_indices = np.delete(random_indices, matches, axis=0)
        
    names = []
    patches = []
    for index in exemplars:
        names.append(patch_dict['names'][index])
        patches.append(patch_dict['patches'][index])
        
    return {'names': names, 'patches': patches}
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('imdir', type=str, metavar='imdir', help='Directory containing volume files')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save the cross sections')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-xy, 1-xz, 2-yz)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=224,
                        help='Size of square image patches. Default 224.')
    parser.add_argument('-hs', '--hash_size', dest='hash_size', type=int, metavar='hash_size', default=8,
                        help='Size of the image hash. Default 8 (assumes crop size of 224).')
    parser.add_argument('-d', '--min_distance', dest='min_distance', type=int, metavar='min_distance', default=12,
                        help='Minimum Hamming distance between hashes to be considered unique. Default 12 (assumes hash size of 8)')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')

    args = parser.parse_args()

    # read in the parser arguments
    imdir = args.imdir
    savedir = args.savedir
    axes = args.axes
    spacing = args.spacing
    crop_size = args.crop_size
    hash_size = args.hash_size
    min_distance = args.min_distance
    processes = args.processes
    
    # check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    # get the list of all volumes (mrc, tif, nrrd, nii.gz, etc.)
    fpaths = glob(os.path.join(imdir, '*'))
    print(f'Found {len(fpaths)} image volumes to process')

    def create_slices(fp):
        # extract the experiment name from the filepath
        # add a special case for .nii.gz files
        if fp[-5:] == 'nii.gz':
            fext = 'nii.gz'
        else:
            fext = fp.split('.')[-1]
            
        exp_name = os.path.basename(fp).split(f'.{fext}')[0]
        
        # check if results have already been generated
        # skip this volume, if so. useful for resuming
        out_path = os.path.join(savedir, exp_name + '.pkl')
        if os.path.isfile(out_path):
            print(f'Already processed {fp}, skipping!')
            return
        
        # try to load the volume, if it's not possible
        # then pass but print        
        try:
            im = sitk.ReadImage(fp)
            
            if len(im.GetSize()) > 3:
                im = im[..., 0]
            
            print('Loaded', fp, im.GetSize())
        except:
            print('Failed to open: ', fp)
            pass
        
        # extract the pixel size from the volume
        # if the z-pixel size is more than 25% different
        # from the x-pixel size, don't slice over orthogonal
        # directions
        pixel_sizes = im.GetSpacing()
        anisotropy = np.abs(pixel_sizes[0] - pixel_sizes[2]) / pixel_sizes[0]
        
        im = sitk.GetArrayFromImage(im)
        assert (im.min() >= 0), 'Negative images not allowed!'
        
        if im.dtype != np.uint8:
            dtype = im.dtype
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            im = im.astype(np.float32) / max_value
            im = (im * 255).astype(np.uint8)
        
        patch_dict = {'names': [], 'patches': [], 'hashes': []}
        for axis in axes:
            # only process xy slices if the volume is anisotropic
            if (anisotropy > 0.25 or 'video' in exp_name.lower()) and (axis != 0):
                continue
                
            # evenly spaced slices
            nmax = im.shape[axis] - 1
            slice_indices = np.arange(0, nmax, spacing, dtype='int')
            zpad = math.ceil(math.log(nmax, 10))
            
            for idx in slice_indices:
                # slice the volume on the proper axis
                if axis == 0:
                    im_slice = im[idx]
                elif axis == 1:
                    im_slice = im[:, idx]
                else:
                    im_slice = im[:, :, idx]
                    
                # crop the image into patches
                patches, hashes, locs = patch_and_hash(im_slice, crop_size, hash_size)

                # appropriate filenames with location
                names = []
                for loc_str in locs:
                    # add the -LOC- to indicate the point of separation between
                    # the dataset name and the slice location information
                    index_str = str(idx).zfill(zpad)
                    patch_loc_str = f'-LOC-{axis}_{index_str}_{loc_str}'
                    names.append(exp_name + patch_loc_str)

                # store results in patch_dict
                patch_dict['names'].extend(names)
                patch_dict['patches'].extend(patches)
                patch_dict['hashes'].extend(hashes)
                
        patch_dict = deduplicate(patch_dict, min_distance)
                
        out_path = os.path.join(savedir, exp_name + '.pkl')
        with open(out_path, 'wb') as handle:
            pickle.dump(patch_dict, handle)
    
    with Pool(processes) as pool:
        pool.map(create_slices, fpaths)
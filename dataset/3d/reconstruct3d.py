"""
Description:
------------

It is assumed that this script will be run after the patchify3d.py and 
classify_patches.py scripts. Errors are certain to occur if that is not the case.

This script takes a directory of filtered 2D image patches and corresponding directories containing
complete 3D reconstructions. It is assumed that at least some of the images in the filtered
directory are cross-sections from the 3D volumes. Matching is performed based on the subdirectory
names and volume names. Multiple volumes may exist for each subdirectory name. This occurs when
the the volumes are actually ROIs from the same dataset. In these cases, it's essential that the
'-ROI-' identifier exists in each subvolume name.

Volumes must be in a format readable by SimpleITK (primarily .mrc, .nrrd, .tif, etc.). THIS
SCRIPT DOES NOT SUPPORT NGFFs. As a rule, such datasets are usually created because of 
their large size. Sparsely sampled ROIs from such NGFF datasets can be downloaded and saved 
in one of the supported formats using the ../scraping/ngff_download.py script.

Example usage:
--------------

python reconstruct3d.py {filtered_dir} \
        -vd {volume_dir1} {volume_dir2} {volume_dir3} \
        -sd {savedir} -nz 224 -p 4 --limit 100
        
Reconstruct a maximum of 100 subvolumes with 224 z-slices from each
dataset represented in {filtered_dir}. Save them in {savedir}, which
will contain a separate subdirectory corresponding to each dataset.

Note1: For generating flipbooks, -nz should always be odd. While even
numbers strictly can be used, they're likely to cause confusion at
annotation time because there isn't a "real" middle slice.

Note2: Z-slices will always be the first dimension in the subvolume 
(this is essential for generating flipbooks).

For help with arguments:
------------------------

python reconstruct3d.py --help
"""

import os, argparse, math
import numpy as np
import SimpleITK as sitk
from skimage import io
from glob import glob
from multiprocessing import Pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('filtered_dir', type=str, metavar='filtered_dir', help='Filtered image directory')
    parser.add_argument('-vd', type=str, dest='volume_dirs', metavar='volume_dirs', nargs='+',
                        help='Directories containing source EM volumes')
    parser.add_argument('-sd', type=str, metavar='savedir', dest='savedir',
                        help='Path to save 3d reconstructions')
    parser.add_argument('-nz', dest='nz', type=int, metavar='nz', default=5,
                        help='Number of z slices in reconstruction')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=32,
                        help='Number of processes to run, more processes run faster but consume memory')
    parser.add_argument('--limit', dest='limit', type=int, metavar='limit',
                        help='Maximum number of reconstructions per volume.')

    args = parser.parse_args()
    filtered_dir = args.filtered_dir
    volume_dirs = args.volume_dirs
    savedir = args.savedir
    numberz = args.nz
    processes = args.processes
    limit = args.limit
    
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    # images stored in subdirectories by source volume
    img_fdirs = glob(os.path.join(filtered_dir, '*'))
    img_fpaths_dict = {}
    for fdir in img_fdirs:
        source_name = fdir.split('/')[-1]
        fnames = np.array([os.path.basename(f) for f in glob(os.path.join(fdir, '*.tiff'))])
        if limit is not None:
            fnames = np.random.choice(fnames, min(limit, len(fnames)), replace=False)
            
        img_fpaths_dict[source_name] = fnames
        
    print(f'Found {len(img_fpaths_dict.keys())} image directories.')    
        
    # volumes may be in multiple directories
    volume_fpaths = []
    for voldir in volume_dirs:
        volume_fpaths.extend(glob(os.path.join(voldir, '*')))
    print(f'Found {len(volume_fpaths)} source volumes.')
    
    def find_children(vol_fpath):
        """
        Finds child images from a source volume.
        """
        # name of volume
        volname = os.path.basename(vol_fpath)
        
        # strip the suffix
        suffix = volname.split('.')[-1]
        assert (suffix in ['mrc', 'nrrd']), \
        f"Found invalid volume file type: {suffix}"
        volname = '.'.join(volname.split('.')[:-1])
        
        # directory with images will be 
        # volname or prefix before -ROI-
        if '-ROI-' in volname:
            dirname = volname.split('-ROI-')[0]
        else:
            dirname = volname
            
        if dirname not in img_fpaths_dict:
            return [], dirname
        
        img_fpaths = img_fpaths_dict[dirname]
        vol_img_fpaths = []
        for fp in img_fpaths:
            if volname in fp:
                vol_img_fpaths.append(fp)
        
        return vol_img_fpaths, dirname
    
    def extract_subvolume(volume, img_fpath):
        """
        Extracts the correct subvolume from the
        full volumetric dataset based on the name
        of a given image which must include the -LOC-
        identifier.
        """
        # extract location of image from filename
        img_fpath = os.path.basename(img_fpath)
        volname, loc = img_fpath.split('-LOC-')
        loc = loc.split('.tiff')[0]
        
        # first the axis
        axis, index, yrange, xrange = loc.split('_')
        
        # convert to integers
        # NOTE: these indices are for a numpy array!
        axis = int(axis)
        index = int(index)
        yslice = slice(*[int(s) for s in yrange.split('-')])
        xslice = slice(*[int(s) for s in xrange.split('-')])
        
        # expand the to include range
        # around index
        span = numberz // 2
        lowz = index - span
        highz = index + span + 1
        
        # pass images that don't have enough context
        if lowz < 0 or highz >= volume.shape[axis]:
            return None, None
        else:
            axis_span = slice(lowz, highz)
            
            if axis == 0:
                subvol = volume[axis_span, yslice, xslice]
            elif axis == 1:
                subvol = volume[yslice, axis_span, xslice]
                subvol = subvol.transpose(1, 0, 2)
            elif axis == 2:
                subvol = volume[yslice, xslice, axis_span]
                subvol = subvol.transpose(2, 0, 1)
            else:
                raise Exception(f'Axis cannot be {axis}, must be in [0, 1, 2]')
                
            subvol_fname = f'{volname}-LOC-{axis}_{lowz}-{highz}_{yrange}_{xrange}'
                
        return subvol, subvol_fname
    
    def create_subvols(vp):
        children, dirname = find_children(vp)
        
        vol_savedir = os.path.join(savedir, dirname)
        if os.path.isdir(vol_savedir):
            print('Skipping', dirname)
            return
        
        if children:
            # load the volume and convert to numpy
            volume = sitk.ReadImage(vp)
            volume = sitk.GetArrayFromImage(volume)
            
            if volume.ndim > 3:
                volume = volume[..., 0]
            
            if np.any(np.array(volume.shape) < numberz):
                raise Exception(f'Subvolume of size {numberz} cannot be created from {vp} with size {volume.shape}')
                
            # directory in which to save subvols 
            # from this volume dataset
            if not os.path.isdir(vol_savedir):
                os.makedirs(vol_savedir, exist_ok=True)
                
            # extract and save subvols
            for child in children:
                subvol, subvol_fname = extract_subvolume(volume, child)
                if subvol_fname is not None:
                    io.imsave(os.path.join(vol_savedir, subvol_fname + '.tif'), 
                              subvol, check_contrast=False)
    
    with Pool(processes) as pool:
        output = pool.map(create_subvols, volume_fpaths)
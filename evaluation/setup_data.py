import os, subprocess, argparse
import SimpleITK as sitk
import numpy as np
from skimage.io import imread, imsave
from glob import glob
from h5py import File

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Setup benchmark dataset directories for NN training and testing')
    parser.add_argument('save_dir', type=str, metavar='save_dir', help='Directory in which to save the benchmark datasets')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    
    #parse the save directory
    args = parse_args()
    save_dir = args['save_dir']

    #run the download_benchmarks.sh script
    command = f'bash download_benchmarks.sh {save_dir}'
    subprocess.call(command.split(' '))

    #first, we need to fix up the lucchi_pp dataset:
    #1. the image and mask filenames are not the same
    #2. mask pixels are 255, should be 1 instead
    #process the train and test datasets separately
    for setname in ['train', 'test']:
        mskpaths = glob(os.path.join(save_dir, f'lucchi_pp/{setname}/masks/*.png'))
        for mp in mskpaths:
            #load the image
            mask = imread(mp)
            
            #convert labelmap values from 255 to 1
            if mask.ndim == 3:
                mask = mask[..., 0]
            
            #background padding values are non-zero
            #dividing by 255 rounds them down to zero
            mask = (mask / 255).astype(np.uint8)
            
            #fix the name to match image names: mask9999.png
            slice_num = mp.split('/')[-1].split('.png')[0]
            
            #save under the new name
            imsave(mp.replace(f'{slice_num}.png', f'mask{slice_num.zfill(4)}.png'), mask, check_contrast=False)
            
            #remove the old file
            os.remove(mp)
            
    #next, fix up the kasthuri_pp dataset:
    #1. mask pixels are 255, should be 1 instead
    #2. masks have 3 channels, should only have 1
    #process the train and test datasets separately
    for setname in ['train', 'test']:
        mskpaths = glob(os.path.join(save_dir, f'kasthuri_pp/{setname}/masks/*.png'))
        for mp in mskpaths:
            #load the image
            mask = imread(mp)
            
            #convert labelmap values from 255 to 1
            #take the first channel only (all 3 are the same anyway)
            if mask.ndim == 3:
                mask = mask[..., 0]
                
            #background padding values are non-zero
            #dividing by 255 rounds them down to zero
            mask = (mask / 255).astype(np.uint8)
            
            #overwrite the mask file
            imsave(mp, mask, check_contrast=False)

    #next, fix up the perez datasets:
    #1. the image and mask filenames are not the same, all have different prefixes
    #so we'll just remove the prefix (*_orig.999.png --> 999.png)
    
    #process all the images in one big group
    perez_fpaths = glob(os.path.join(save_dir, f'perez/*/*/*/*.png')) #e.g. perez/mito/train/images/*.png
    for fp in perez_fpaths:
        orig_name = fp.split('/')[-1]
        prefix = orig_name.split('.')[0]
        new_name = orig_name.replace(f'{prefix}.', '') #remove the prefix and the trailing dot
        os.rename(fp, fp.replace(orig_name, new_name))
        
    #next, the guay dataset:
    #1. image volumes are 16-bit signed pixels, convert them to 8-bit unsigned
    #2. mask volumes are 16-bit unsigned pixels, convert them to 8-bit unsigned
    #3. mask volumes have a different name that image volumes change '-labels' to '-images'
    #4. slice cross sections to make 2d versions of the train, validation, and test sets
    guay_impaths = glob(os.path.join(save_dir, f'guay/3d/*/images/*.tif')) #e.g. guay/3d/train/images/*.tif
    uint16_max = 2 ** 16 - 1
    int16_min = -(2 ** 15)
    for ip in guay_impaths:
        #load the volume
        vol = sitk.ReadImage(ip)
        
        #convert to numpy
        vol = sitk.GetArrayFromImage(vol)
        
        #convert the volume to float
        vol = vol.astype('float')
        
        #subtract int16_min, divide by uint16_max, multiply by 255
        vol = ((vol - int16_min) / uint16_max) * 255
        
        #convert to uint8
        vol = vol.astype(np.uint8)
        
        #create a new volume
        sitk.WriteImage(sitk.GetImageFromArray(vol), ip.replace('.tif', '.nrrd'))
        
        #remove the original volume
        os.remove(ip)
        
    guay_mskpaths = glob(os.path.join(save_dir, f'guay/3d/*/masks/*.tif')) #e.g. guay/3d/test/masks/*.tif
    for mp in guay_mskpaths:
        #load the volume
        vol = sitk.ReadImage(mp)
        
        #convert to uint8
        vol = sitk.Cast(vol, sitk.sitkUInt8)
        
        #replace '-labels' with '-images' in the filename
        sitk.WriteImage(vol, mp.replace('-labels.tif', '-images.nrrd'))
        
        #remove the original volume
        os.remove(mp)
        
    #now we can slice the volumes into cross sections
    #sort the impaths and mskpaths just to ensure that they line up
    guay_impaths = np.sort(glob(os.path.join(save_dir, f'guay/3d/*/images/*.nrrd')))
    guay_mskpaths = np.sort(glob(os.path.join(save_dir, f'guay/3d/*/masks/*.nrrd')))
    for ip, mp in zip(guay_impaths, guay_mskpaths):
        assert(ip.replace('/images/', '/blank/') == mp.replace('/masks/', '/blank/')), "Image and mask volumes are not aligned!"
        #call the create_slices.py script to save results in the guay/2d folder
        setname = ip.split('/')[-3] #.../guay/3d/train/images/train-images.tif --> train
        slice_dir = os.path.join(save_dir, f'guay/2d/{setname}/')
        command = f'python create_slices.py {ip} {mp} {slice_dir} -a 2 -s 1'
        subprocess.call(command.split(' '))
        
    
    #now onto UroCell:
    #1. mito and lyso labelmaps, are separate, we need to combine them into a single mask volume
    #2. slice cross sections to make 2d versions of the train and test set
    
    #get paths of the mito and lyso labelmap volumes
    lysopaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/lyso/*.nii.gz'))) #e.g. urocell/3d/train/lyso/*.nii.gz
    mitopaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/mito/*.nii.gz'))) #e.g. urocell/3d/train/mito/*.nii.gz
    for lp, mp in zip(lysopaths, mitopaths):
        assert(lp.replace('/lyso/', '/mito/') == mp), "Lyso and mito label volumes are not aligned!"
        
        #load the volumes
        lyso = sitk.ReadImage(lp)
        mito = sitk.ReadImage(mp)
        
        #add them together into a single label volume
        #such that 1 == lyso and 2 == mito
        labelmap = lyso + 2 * mito
        
        #make sure the datatype is uint8
        labelmap = sitk.Cast(labelmap, sitk.sitkUInt8)
        
        #save the result
        sitk.WriteImage(labelmap, lp.replace('/lyso/', '/masks/'))
        
    #now we're ready to slice into cross sections
    #get impaths and mskpaths
    urocell_impaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/images/*.nii.gz'))) #e.g. urocell/3d/test/images/*.nii.gz
    urocell_mskpaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/masks/*.nii.gz'))) #e.g. urocell/3d/train/masks/*.nii.gz
    for ip, mp in zip(urocell_impaths, urocell_mskpaths):
        assert(ip.replace('/images/', '/blank/') == mp.replace('/masks/', '/blank/')), "Image and mask volumes are not aligned!"
        #call the create_slices.py script to save results in the urocell/2d folder
        setname = ip.split('/')[-3] #.../urocell/3d/train/images/fib1-4-3-0.nii.gz --> train
        slice_dir = os.path.join(save_dir, f'urocell/2d/{setname}/')
        command = f'python create_slices.py {ip} {mp} {slice_dir} -a 0 1 2 -s 1'
        subprocess.call(command.split(' '))
        

    
    #next we're going to handle CREMI
    #1. extract image volumes and labelmap volumes from .hdf files
    #2. binarize the synaptic cleft labelmaps and convert to uint8
    #3. Slice into cross sections
    cremi_hdfpaths = glob(os.path.join(save_dir, f'cremi/3d/*/*.hdf')) #e.g. cremi/3d/test/*.hdf
    for hdfp in cremi_hdfpaths:
        #extract the setname (train or test)
        setname = hdfp.split('/')[-2] #e.g. cremi/3d/test/*.hdf --> test
        
        #load the hdf file
        dataset = File(hdfp, mode='r')['volumes']
        
        #get the image vol, which is already uint8
        imvol = dataset['raw'].__array__()
        
        #get the mask volume which needs to be binarized
        #and inverted and saved as uint8
        mskvol = dataset['labels']['clefts'].__array__()
        mskvol = np.invert(mskvol == 0xffffffffffffffff).astype(np.uint8)
        
        #save both image and mask as .nrrd files
        imvol = sitk.GetImageFromArray(imvol)
        new_path = hdfp.replace('.hdf', '.nrrd').replace(f'/{setname}/', f'/{setname}/images/')
        sitk.WriteImage(imvol, new_path)
        
        mskvol = sitk.GetImageFromArray(mskvol)
        new_path = hdfp.replace('.hdf', '.nrrd').replace(f'/{setname}/', f'/{setname}/masks/')
        sitk.WriteImage(mskvol, new_path)
        
        #remove the hdf file
        os.remove(hdfp)

    #now we can slice the volumes into cross sections
    #sort the impaths and mskpaths just to ensure that they line up
    cremi_impaths = np.sort(glob(os.path.join(save_dir, f'cremi/3d/*/images/*.nrrd')))
    cremi_mskpaths = np.sort(glob(os.path.join(save_dir, f'cremi/3d/*/masks/*.nrrd')))
    for ip, mp in zip(cremi_impaths, cremi_mskpaths):
        assert(ip.replace('/images/', '/blank/') == mp.replace('/masks/', '/blank/')), "Image and mask volumes are not aligned!"
        #call the create_slices.py script to save results in the cremi/2d folder
        setname = ip.split('/')[-3] #.../cremi/3d/train/images/sampleA.nrrd --> train
        slice_dir = os.path.join(save_dir, f'cremi/2d/{setname}/')
        command = f'python create_slices.py {ip} {mp} {slice_dir} -a 2 -s 1'
        subprocess.call(command.split(' '))

    #finally, let's make the All Mitochondria dataset
    #from perez mito, lucchi, kasthuri, urocell, and guay
    #1. create directories
    #2. crop images from each benchmark into 256x256 patches
    #this is needed so that datasets like Kasthuri with 85 large images
    #have a similar number of patches relative to a dataset like
    #UroCell with 3200 small images
    #3. A portion of patches from the Kasthuri dataset contain
    #nothing by background padding; we want to remove them
    
    #make the directories
    #no need for a test directory because we use the
    #source benchmarks' test sets
    if not os.path.isdir(os.path.join(save_dir, 'all_mito')):
        os.mkdir(os.path.join(save_dir, 'all_mito'))
        os.mkdir(os.path.join(save_dir, 'all_mito/train'))
        os.mkdir(os.path.join(save_dir, 'all_mito/train/images'))
        os.mkdir(os.path.join(save_dir, 'all_mito/train/masks'))
    
    #crop images in 256x256 patches from their sources directories
    benchmarks = ['perez/mito', 'lucchi_pp', 'kasthuri_pp', 'urocell/2d', 'guay/2d']
    benchmark_mito_labels = [1, 1, 1, 2, 2]
    for l, bmk in zip(benchmark_mito_labels, benchmarks):
        command = f'python create_patches.py {save_dir}/{bmk}/train/images/ {save_dir}/all_mito/train/images/'
        subprocess.call(command.split(' '))
        
        command = f'python create_patches.py {save_dir}/{bmk}/train/masks/ {save_dir}/all_mito/train/masks/ -l {l}'
        subprocess.call(command.split(' '))

    #remove any blank images and their corresponding masks
    impaths = np.sort(glob(os.path.join(save_dir, f'all_mito/train/images/*')))
    mskpaths = np.sort(glob(os.path.join(save_dir, f'all_mito/train/masks/*')))
    for ip, mp in zip(impaths, mskpaths):
        assert(ip.replace('/images/', '/blank/') == mp.replace('/masks/', '/blank/')), "Image and mask file paths are not aligned!"
        
        #load the image file
        image = imread(ip)
        
        #if more than 95% of the image is black padding
        #remove the image and it's corresponding mask
        thr = (256 ** 2) * 0.05
        if (image > 0).sum() < thr:
            os.remove(ip)
            os.remove(mp)
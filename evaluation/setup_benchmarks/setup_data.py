import os, subprocess, argparse, shutil, yaml
import SimpleITK as sitk
import numpy as np
from skimage.io import imread, imsave
from glob import glob
from skimage.transform import resize
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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    #run the download_benchmarks.sh script
    download_script = os.path.join(script_dir, 'download_benchmarks.sh')
    create_slices_script = os.path.join(script_dir, 'create_slices.py')
    create_patches_script = os.path.join(script_dir, 'create_patches.py')

    command = f'bash {download_script} {save_dir}'
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
    for ip in guay_impaths:
        #load the volume
        vol = sitk.ReadImage(ip)
        
        #convert to numpy
        vol = sitk.GetArrayFromImage(vol)
        
        #convert the volume to float
        vol = vol.astype('float')
        
        #subtract min, divide by max, multiply by 255
        vol -= vol.min()
        vol /= vol.max()
        vol *= 255
        
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
        command = f'python {create_slices_script} {ip} {mp} {slice_dir} -a 2 -s 1'
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
        
        #for two of the volumes we need to crop out some
        #regions with missing data
        if lp.split('/')[-1] == 'fib1-0-0-0.nii.gz':
            labelmap = labelmap[:, 12:]
        elif lp.split('/')[-1] == 'fib1-1-0-3.nii.gz':
            labelmap = labelmap[:, 54:]
        
        #save the result
        sitk.WriteImage(labelmap, lp.replace('/lyso/', '/masks/'))
        
    #now we're ready to slice into cross sections
    #get impaths and mskpaths
    urocell_impaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/images/*.nii.gz'))) #e.g. urocell/3d/test/images/*.nii.gz
    urocell_mskpaths = np.sort(glob(os.path.join(save_dir, f'urocell/3d/*/masks/*.nii.gz'))) #e.g. urocell/3d/train/masks/*.nii.gz
    for ip, mp in zip(urocell_impaths, urocell_mskpaths):
        assert(ip.replace('/images/', '/blank/') == mp.replace('/masks/', '/blank/')), "Image and mask volumes are not aligned!"
        
        #convert image from float to uint8
        image = sitk.Cast(sitk.ReadImage(ip), sitk.sitkUInt8)
        
        #for two of the volumes we need to crop out some
        #regions with missing data
        if ip.split('/')[-1] == 'fib1-0-0-0.nii.gz':
            image = image[:, 12:]
        elif ip.split('/')[-1] == 'fib1-1-0-3.nii.gz':
            image = image[:, 54:]
        
        sitk.WriteImage(image, ip)
        
        #call the create_slices.py script to save results in the urocell/2d folder
        setname = ip.split('/')[-3] #.../urocell/3d/train/images/fib1-4-3-0.nii.gz --> train
        slice_dir = os.path.join(save_dir, f'urocell/2d/{setname}/')
        command = f'python {create_slices_script} {ip} {mp} {slice_dir} -a 0 1 2 -s 1'
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
        command = f'python {create_slices_script} {ip} {mp} {slice_dir} -a 2 -s 1'
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
    if not os.path.isdir(os.path.join(save_dir, 'all_mito')):
        os.mkdir(os.path.join(save_dir, 'all_mito'))
        os.makedirs(os.path.join(save_dir, 'all_mito/train/images'))
        os.mkdir(os.path.join(save_dir, 'all_mito/train/masks'))
        os.makedirs(os.path.join(save_dir, 'all_mito/test/2d/'))
        os.mkdir(os.path.join(save_dir, 'all_mito/test/3d/'))

    #crop images in 256x256 patches from their sources directories
    benchmarks = ['perez/mito', 'lucchi_pp', 'kasthuri_pp', 'urocell/2d', 'guay/2d']
    benchmark_mito_labels = [1, 1, 1, 2, 2]
    for l, bmk in zip(benchmark_mito_labels, benchmarks):
        command = f'python {create_patches_script} {save_dir}/{bmk}/train/images/ {save_dir}/all_mito/train/images/'
        subprocess.call(command.split(' '))
        
        command = f'python {create_patches_script} {save_dir}/{bmk}/train/masks/ {save_dir}/all_mito/train/masks/ -l {l}'
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
        
    #copy the test images from 2d datasets into the 2d test directory
    benchmarks = ['perez/mito', 'lucchi_pp', 'kasthuri_pp']
    benchmark_mito_labels = [1, 1, 1]
    for l, bmk in zip(benchmark_mito_labels, benchmarks):
        #glob all the images in the given test directory
        impaths = glob(os.path.join(save_dir, f'{bmk}/test/images/*'))
        bmk = 'perez_mito' if bmk == 'perez/mito' else bmk
        im_dst_dir = os.path.join(save_dir, f'all_mito/test/2d/{bmk}/images/')
        msk_dst_dir = os.path.join(save_dir, f'all_mito/test/2d/{bmk}/masks/')
        os.makedirs(im_dst_dir)
        os.makedirs(msk_dst_dir)
        
        for imp in impaths:
            fname = imp.split('/')[-1]
            shutil.copy(imp, im_dst_dir + fname)
            
            #do the same for the corresponding mask
            imp = imp.replace('/images/', '/masks/')
            shutil.copy(imp, msk_dst_dir + fname)
    

    #now copy only the mito label from the 3d benchmarks
    benchmarks = ['urocell/3d', 'guay/3d']
    benchmark_mito_labels = [2, 2]
    for l, bmk in zip(benchmark_mito_labels, benchmarks):
        #glob all the images in the given test directory
        impaths = glob(os.path.join(save_dir, f'{bmk}/test/images/*'))
        
        bn_name = bmk.split('/')[0]
        im_dst_dir = os.path.join(save_dir, f'all_mito/test/3d/{bn_name}/images/')
        msk_dst_dir = os.path.join(save_dir, f'all_mito/test/3d/{bn_name}/masks/')
        os.makedirs(im_dst_dir)
        os.makedirs(msk_dst_dir)
        
        for imp in impaths:
            #copy the image volume directly
            fname = imp.split('/')[-1]
            shutil.copy(imp, im_dst_dir + fname)
            
            #open the mask volume with simpleitk
            #and remove everything but the mito label
            imp = imp.replace('/images/', '/masks/')
            if bn_name == 'urocell':
                vol = sitk.ReadImage(imp)
                labelmap = vol == 2
                mito_vol = sitk.Cast(labelmap, sitk.sitkUInt8)
            else:
                #keep labels 0, 1, 2 for guay (label 1 is the mask in which 
                #the ground truth is defined)
                #load the volumes
                vol = sitk.ReadImage(imp)
                cell = vol == 1
                other = vol > 2
                mito = vol == 2
                #make the mito label 1 and the cell label 2
                labelmap = mito + 2 * (cell + other)
                mito_vol = sitk.Cast(labelmap, sitk.sitkUInt8)
            
            #save the result
            sitk.WriteImage(mito_vol, msk_dst_dir + fname)       
    
    #overwrite the data and test directories in each benchmarks' yaml file
    benchmarks = ['all_mito', 'cremi', 'guay', 'kasthuri_pp', 'lucchi_pp', 
                  'perez_lyso', 'perez_mito', 'perez_nuclei', 'perez_nucleoli', 'urocell']
    data_dirs = ['all_mito/', 'cremi/2d/', 'guay/2d/', 'kasthuri_pp/', 'lucchi_pp/',
                 'perez/lyso/', 'perez/mito/', 'perez/nuclei/', 'perez/nucleoli/', 'urocell/2d/']
    test_dirs = ['all_mito/test/', 'cremi/3d/test/', 'guay/3d/test/', 'kasthuri_pp/test/', 'lucchi_pp/test/', 'perez/lyso/test/',
                 'perez/mito/test/', 'perez/nuclei/test/', 'perez/nucleoli/test/', 'urocell/3d/test/']
    
    config_dir = os.path.join(script_dir, '../benchmark_configs/')
    
    #overwrite the data_dir and test_dir lines to match
    #the directories created in this setup script
    for bmk, dd, td in zip(benchmarks, data_dirs, test_dirs):
        with open(os.path.join(config_dir, f'{bmk}.yaml'), mode='r') as f:
            lines = f.read().splitlines()
            data_dir_ln = -1
            test_dir_ln = -1
            test_dir_name = 'test_dir'
            for ix, l in enumerate(lines):
                if l.startswith('data_dir:'):
                    data_dir_ln = ix
                elif l.startswith('test_dir2d:'):
                    test_dir_ln = ix
                    test_dir_name = 'test_dir2d'
                elif l.startswith('test_dir3d:'):
                    test_dir_ln = ix
                    test_dir_name = 'test_dir3d'
                elif l.startswith('test_dir:'):
                    test_dir_ln = ix
                    test_dir_name = 'test_dir'

            lines[data_dir_ln] = 'data_dir: ' + '"' + os.path.join(save_dir, dd) + '"'
            lines[test_dir_ln] = f'{test_dir_name}: ' + '"' + os.path.join(save_dir, td) + '"'
            lines = '\n'.join(lines)

        with open(os.path.join(config_dir, f'{bmk}.yaml'), mode='w') as f:
            f.write(lines)
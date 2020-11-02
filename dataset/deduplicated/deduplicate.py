"""
Description:
------------

It is assumed that this script will be run after the cross_section.py and 
crop_patches.py scripts. Errors are certain to occur if that is not the case.

This script takes a directory containing image patches and their corresponding
hashes and performs deduplication of patches from within the same dataset. The resulting
array of deduplicated patches is stored is a list of filepaths in the given savedir with
the name deduplicated_fpaths.npz

Example usage:
--------------

python deduplicate.py {patchdir} {savedir} --min_distance 12 --processes 32

For help with arguments:
------------------------

python deduplicate.py --help
"""

import os, argparse
import numpy as np
import dask.array as da
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('patchdir', type=str, metavar='patchdir', help='Directory containing image patches and hashes')
    parser.add_argument('savedir', type=str, metavar='savedir', 
                        help='Path to save array containing the paths of exemplar images')
    parser.add_argument('-d', '--min_distance', dest='min_distance', type=int, metavar='min_distance', default=12,
                        help='Minimum Hamming distance between hashes to be considered unique')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=32,
                        help='Number of processes to run, more processes run faster but consume memory')
    

    #parse the arguments
    args = parser.parse_args()
    patchdir = args.patchdir
    savedir = args.savedir
    min_distance = args.min_distance
    processes = args.processes
    
    #to avoid running this long script only to get a nasty error
    #let's make sure that the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    #load all the image filenames and save them as a dask array. 
    #With over 5 million strings in each array, the memory
    #requirements become fairly hefty. The dask array
    #has slightly slower I/O but saves a considerable amount
    #of memory. In case this deduplication script has been run before, we'll
    #check to see if the dask array already exists
    da_impaths_path = os.path.join(patchdir, 'unfiltered_fpaths.npz')
    if not os.path.isfile(da_impaths_path):
        impaths = np.sort(glob(os.path.join(patchdir, '*.tiff')))
        impaths = da.from_array(impaths)
        da.to_npy_stack(da_impaths_path, impaths)
        del impaths

    #load the dask array of impaths
    impaths = da.from_npy_stack(da_impaths_path)
    print(f'Found {len(impaths)} images to deduplicate')

    def get_dataset_name(imf):
        #function to extract the name of a dataset from the patch image file path
        #in the cross_section.py script we added the handy -LOC- indicator to
        #easily identify the source dataset from location information
        return imf.split('/')[-1].split('-LOC-')[0]

    #extract the set of unique dataset names from all the impaths
    with Pool(processes) as pool:
        datasets = np.array(pool.map(get_dataset_name, impaths.compute()))

    #because we sorted the impaths, we know that all images from the
    #same dataset will be grouped together. therefore, we only need
    #to know the index of the first instance of a unique dataset name
    #in order to get the indices of all the patches from that dataset
    unq_datasets, indices = np.unique(datasets, return_index=True)
    
    #we can delete the datasets array now
    del datasets
    
    #add the last index for impaths such that we have complete intervals
    indices = np.append(indices, len(impaths))

    #make groups of image patches by source dataset
    groups_impaths = []
    for si, ei in zip(indices[:-1], indices[1:]):
        #have to call .compute() for a dask array
        groups_impaths.append(impaths[si:ei].compute())
        
    #now we can delete the impaths dask array
    del impaths

    #sanity check that we have the same number of
    #unique datasets and impath groups
    assert(len(unq_datasets) == len(groups_impaths))

    #define the function for deduplication of a group of image paths
    def group_dedupe(args):
        #two arguments are the unique dataset name and the filepaths of
        #the patches that belong to that dataset
        dataset_name, impaths = args
        
        #check if we already processes this dataset name, if so
        #then we can skip it
        exemplar_fpath = os.path.join(savedir, f'{dataset_name}_exemplars.npy')
        if os.path.isfile(exemplar_path):
            return None
        
        #randomly permute the impaths such that we'll have random ordering
        impaths = np.random.permutation(impaths)
        
        #requires that hash array and tiff image are in the 
        #same directory (which is how crop_patches.py is setup)
        hashes = np.array([np.load(ip.replace('.tiff', '.npy')).ravel() for ip in impaths])

        #make a list of exemplar images to keep
        exemplars = []
        impaths = np.array(impaths)
        
        #loop through the hashes and assign images to sets of near duplicates
        #until all of the hashes are exhausted
        while len(hashes) > 0:
            #the reference hash is the first one in the list
            #of remaining hashes
            ref_hash = hashes[0]
            
            #a match has Hamming distance less than min_distance
            matches = np.where(np.logical_xor(ref_hash, hashes).sum(1) <= min_distance)[0]
            
            #choose the first match as the exemplar and add
            #it's filepath to the list. this is random because we
            #permuted the paths earlier. a different image could be
            #chosen on another run of this script
            exemplars.append(impaths[matches[0]])
            
            #remove all the matched images from both hashes and impaths
            hashes = np.delete(hashes, matches, axis=0)
            impaths = np.delete(impaths, matches, axis=0)
            
        #because this script can take a long time to complete, let's save checkpoint
        #results for each dataset when it's finished with deduplication, then we have
        #the option to resume later on
        np.save(exemplar_fpath, np.array(exemplars))

    #run the dataset level deduplication on multiple groups at once
    #results for each group are saved in separate .npy files, if the
    #.npy file already exists, then it will be skipped. This makes it
    #easier to add new datasets to the existing directory structure
    with Pool(processes) as pool:
        pool.map(group_dedupe, list(zip(unq_datasets, groups_impaths)))

    #now that all the patches from individual datasets are deduplicated,
    #we'll combine all the separate .npy arrays into a single dask array and save it
    exemplar_fpaths = glob(os.path.join(save_dir, '_exemplars.npy'))
    deduplicated_fpaths = np.concatenate([np.load(fp) for fp in exemplar_fpaths])
    
    #convert to dask and save
    deduplicated_fpaths = da.from_array(deduplicated_fpaths)
    da.to_npy_stack(os.path.join(save_dir, 'deduplicated_fpaths.npz'), deduplicated_fpaths)
    
    #print the total number of deduplicated patches
    print(f'{len(deduplicated_fpaths)} patches remaining after deduplication.')
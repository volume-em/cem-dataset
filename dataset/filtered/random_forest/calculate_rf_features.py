"""
Description:
------------

It is assumed that a deduplicated dataset has already by created
from deduplicate.py. This script calculates four image features with which to train a random
forest model for filtering "good" and "bad" quality patches. 

The impaths_file argument expects a dask array file containing fpaths to tiff images. For
example, the deduplicated_fpaths.npz file created by deduplicate.py. Results are saved
in the given savedir with the same name as the input dask array file, but with the 
suffix _rf_features.npy instead.

Example usage:
--------------

python calculate_rf_features.py {path}/deduplicated_fpaths.npz {savedir}

For help with arguments:
------------------------

python calculate_rf_features.py --help
"""

import os
import numpy as np
import dask.array as da
from multiprocessing import Pool
from glob import glob

from skimage import io
from skimage.morphology import square
from skimage.feature import canny, local_binary_pattern
from skimage.filters.rank import entropy, geometric_mean

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('impaths_file', type=str, metavar='impaths_file', 
                        help='Path to .npz dask array file containing patch filepaths (for example output of deduplicate.py)')
    parser.add_argument('savedir', type=str, metavar='savedir', 
                        help='Directory in which to save the array of calculated features')

    #parse the arguments
    args = parser.parse_args()
    impaths_file = args.impaths_file
    savedir = args.savedir
    
    #make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    #load the dask array
    impaths = da.from_npy_stack(impaths_file)

    def calculate_features(imfile):
        #these features are based on validation results that showed these four
        #features were the most important out of a set of ~30 features
        #for the classification of "good" and "bad" patches. although
        #restricting ourselves to only these 4 features may result in
        #slightly lower prediction accuracy, the time it takes to calculate
        #30 features on ~1 million images is excessive
        #call .compute() for dask array element
        image = io.imread(imfile.compute())

        features = []
        #first lbp stdev
        features.append(local_binary_pattern(image, 8, 8).std())

        #second median of geo. mean
        features.append(np.median(geometric_mean(image, square(11))))

        #third stdev of entropy
        features.append(entropy(image, square(7)).std())

        #fourth mean of the canny filter
        features.append(canny(image, sigma=1).mean())
        return features

    with Pool(32) as pool:
        features = np.array(list(pool.map(calculate_features, impaths)))

    #save the features as a .npy array in the save directory
    #first, get the name of the list of impaths
    source_name = impaths_file.split('/').split('.npz')[0]
    save_path = os.path.join(savedir, f'{source_name}_rf_features.npy')
    np.save(save_path, features)
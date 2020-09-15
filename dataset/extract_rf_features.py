import os
import numpy as np
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
from time import time

from skimage import io
from skimage.morphology import square
from skimage.feature import canny, local_binary_pattern
from skimage.filters.rank import entropy, geometric_mean

start = time()
time_limit = 12 * (3600) - (10 * 60) #give a 10 minute buffer for saving

def calc_features(imfile):
    if time() - start > time_limit:
        return None
    else:
        image = io.imread(imfile)
        features = []
        #first lbp std
        features.append(local_binary_pattern(image, 8, 8).std())
        #second median of geo. mean
        features.append(np.median(geometric_mean(image, square(11))))
        #third stdev of entropy
        features.append(entropy(image, square(7)).std())
        #fourth mean of the canny filter
        features.append(canny(image, sigma=1).mean())
        return features

imfiles = np.load('/data/IASEM/conradrw/data/images224_fpaths_qsf.npy')

with Pool(32) as pool:
    #features = np.array(list(tqdm(pool.imap(calc_features, imfiles), total=len(imfiles))))
    features = np.array(list(pool.map(calc_features, imfiles)))
    
#print(features.shape)
np.save('/data/IASEM/conradrw/data/images224_fpaths_qsf_rf_features.npy', features)

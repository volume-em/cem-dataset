import os
import imagehash
import numpy as np
from PIL import Image
from glob import glob
from multiprocessing import Pool
from time import time
from tqdm import tqdm

print('imported modules')
imfiles = np.load('/data/IASEM/conradrw/data/images224_fpaths.npy')
#imfiles = np.random.permutation(np.load('/data/IASEM/conradrw/data/images_224_fnames_round8.npy'))
print(f'Found {len(imfiles)} image files')

#create groups based on the dataset name
def dataset_name(imf):
    return '_'.join(imf.split('/')[-1].split('_')[:-4])

with Pool(os.cpu_count()) as pool:
    datasets = list(tqdm(pool.imap(dataset_name, imfiles), total=len(imfiles)))

unq_datasets, indices = np.unique(datasets, return_index=True)
indices = np.append(indices, len(imfiles))

groups_imfiles = []
for si, ei in zip(indices[:-1], indices[1:]):
    groups_imfiles.append(imfiles[si:ei])
    
#for ds in unq_datasets:
#    print(ds)
#    groups_imfiles.append(imfiles[np.where(np.core.defchararray.find(imfiles, ds) > -1)[0]])
    
#groups_imfiles = []
#starts = np.arange(0, len(imfiles), step=51800)
#ends = starts + 51800
#ends[-1] = len(imfiles) - 1

#for s,e in zip(starts, ends):
#    groups_imfiles.append(imfiles[s:e])
    
#assert(len(unq_datasets) == len(groups_imfiles))

def group_dedupe(group_imfiles):
    print(len(group_imfiles))
    s = time()
    group_hashes = []
    for imf in group_imfiles:
        hashf = imf.replace('images224', 'hashes224').replace('.tiff', '.npy')
        group_hashes.append(np.load(hashf).ravel())
    group_hashes = np.array(group_hashes)
    
    thr = 12
    keep = []
    group_imfiles = np.array(group_imfiles)
    
    while len(group_hashes) > 0:
        #print(group_hashes.shape)
        ref_hash = group_hashes[0]
        matches = np.where(np.logical_xor(ref_hash, group_hashes).sum(1) <= thr)[0]
        keep.append(group_imfiles[matches[0]])
        group_hashes = np.delete(group_hashes, matches, axis=0)
        group_imfiles = np.delete(group_imfiles, matches, axis=0)
        
    print(f'Keeping {len(keep)} after {time() - s} secs')
    return keep

with Pool(os.cpu_count()) as pool:
    keep_fnames = list(pool.map(group_dedupe, groups_imfiles))
    
np.save('/data/IASEM/conradrw/data/images224_fpaths_dedupe.npy', np.concatenate(keep_fnames))

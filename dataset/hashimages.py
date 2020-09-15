import os
import numpy as np
from PIL import Image
import imagehash
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm

def save_hash(imf):
    ftarg = imf.replace('images_224', 'hashes_224')
    np.save(ftarg, imagehash.dhash(Image.open(imf), hash_size=8).hash)

imfiles = glob('/data/IASEM/conradrw/data/images_224/*.tiff')
print(f'Found {len(imfiles)} tiff images')

"""
subset = ['151215_Westlake_SampB_AGAlB4invBV2',
     '160102_Westlake_SampA_FOAlB4invBV2',
     '160129_Westlake_SampA_3V2AlB4invBV2',
     '160202_Westlake_SampA_FO_2ndcellAlB4invBV2',
     '170713_Westlake_TipTube_SampA_3LAlB3invBV2',
     '170814_Westlake_SampB_3Hr_rd5_7MAlB3invBV2',
     '170822_0HRNotbinnedinvBV2',
     '170823_Westlake_SampleA_rd5_0HR_2JAlB3invBV2',
     '180116_Westlake_SampleB_3Hr_9SAlB3invBV2',
     '180305_Westlake_Ru-Red_SampB_BJAlB3invBV2',
     '180312_Westalake_Ru-Red_SampB_6LAlB3invBV2',
     '180314_Westlake_SIMCLEM_SampA_AOAlB3invBV2',
     '180315_Westalake_Ru_Red_SampB_3LAlB3invBV2',
     '180613_West_SIMCLEM_SampA_9KAlB3invBV2',
     '180618_West_SIMCLEM_SampA_7M007AlB3invBV2',
     '180620_West_SIMCLEM_SampA_5IAlB3invBV2',
     '180921_Westlake_SIM_CLEM_siEHD1_9iALB3invBV2',
     '180926_Westlake_SIM_CLEM_Rab8_AHAlB3invBV2',
     '181001_Westlake_SIM_CLEM_siEHD1_2KAlB3invBV2',
     '190723_Westlake_EM8398B_SampCIG_6W_pos1AlB3invBV2',
     'cali2018_mouse_1',
     'cali2018_mouse_2',
     'cil_micro8643_stack',
     'guay2020_eval-images',
     'guay2020_test-images',
     'guay2020_train-images',
     'lucchi2012_testing',
     'lucchi2012_training',
     'segem2015_1028',
     'segem2015_1033',
     'segem2015_1123',
     'segem2015_59']
    
subset_imfiles = []
for imf in tqdm(imfiles):
    if len(np.where(np.core.defchararray.find(np.array(subset), '_'.join(imf.split('/')[-1].split('_')[:-4])) > -1)[0]) == 1:
        subset_imfiles.append(imf)

imfiles = np.array(subset_imfiles)
print(f'Using subset of {len(imfiles)} images')
"""

with Pool(32) as pool:
    hashes = list(tqdm(pool.imap(save_hash, imfiles), total=len(imfiles)))
    #hashes = list(pool.map(save_hash, imfiles))
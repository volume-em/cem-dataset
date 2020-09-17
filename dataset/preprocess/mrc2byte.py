import os
import subprocess
from glob import glob
from tqdm import tqdm

fnames = glob('./moco_data_add/*.mrc')
print('Found {} mrc files'.format(len(fnames)))
FNULL = open(os.devnull, 'w')

for fn in tqdm(fnames):
    command = ['newstack', fn, fn, '-by', '0']
    subprocess.call(command, stdout=FNULL, stderr=subprocess.STDOUT)
    os.remove(fn + '~')
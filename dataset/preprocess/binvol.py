"""
Description:
------------

This script is used to bin .mrc files by a factor of 2. This is a standard
preprocessing step to lower the resolution from under 10 nm to 10-20 nm range.

For help downloading and installing IMOD, see:
https://bio3d.colorado.edu/imod/

Example usage:
--------------

python binvol.py {mrcdir} {--inplace}

For help with arguments:
------------------------

python binvol.py --help

"""

import os, argparse
import subprocess
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mrcdir', type=str, help='Directory containing mrc image volumes')
    parser.add_argument('--inplace', action='store_true', 
                        help='If passed, original mrcs will be permanently deleted in favor of the binned data.')
    
    args = parser.parse_args()
    
    #read in the argument
    mrcdir = args.mrcdir
    inplace = args.inplace

    #gather the mrc filepaths
    fnames = glob(os.path.join(mrcdir, '*.mrc'))
    print('Found {} mrc files to bin'.format(len(fnames)))
    FNULL = open(os.devnull, 'w')
    
    for fn in fnames:
        #create the IMOD command and run it
        command = ['binvol', fn, fn.replace('.mrc', 'BV2.mrc')]
        subprocess.call(command, stdout=FNULL, stderr=subprocess.STDOUT)
        
        if inplace:
            os.remove(fn)
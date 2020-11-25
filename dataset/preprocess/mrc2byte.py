"""
Description:
------------

Throughout this repository we use SimpleITK to load image volumes. The MRC
file format can sometimes cause issues when the files are saved with signed
bytes. To prevent errors in which images are cliiped from 0-127, it is necessary
to make the mrc volumes unsigned byte type. This script takes a directory
containing mrc files and performs that conversion using IMOD.

For help downloading and installing IMOD, see:
https://bio3d.colorado.edu/imod/

Example usage:
--------------

python mrc2byte.py {mrcdir}

For help with arguments:
------------------------

python mrc2byte.py --help

"""

import os, argparse
import subprocess
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mrcdir', type=str, help='Directory containing mrc image volumes')
    args = parser.parse_args()
    
    #read in the argument
    mrcdir = args.mrcdir

    #gather the mrc filepaths
    fnames = glob(os.path.join(mrcdir, '*.mrc'))
    print('Found {} mrc files'.format(len(fnames)))
    FNULL = open(os.devnull, 'w')
    
    for fn in fnames:
        #create the IMOD command and run it
        command = ['newstack', fn, fn, '-by', '0']
        subprocess.call(command, stdout=FNULL, stderr=subprocess.STDOUT)
        
        #IMOD won't overwrite the old file
        #instead it renames it with a '~' at
        #the end. here we remove that old file
        os.remove(fn + '~')
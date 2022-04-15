"""
Description:
------------

Simple script for converting video files (mp4 and avi) into
nrrd image volumes.

Example usage:
--------------

python vid2stack.py {viddir}

For help with arguments:
------------------------

python vid2stack.py --help

"""

import os, argparse
import cv2
import numpy as np
import SimpleITK as sitk
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('viddir', type=str, help='Directory containing video files: avi or mp4')
    args = parser.parse_args()

    viddir = args.viddir
    
    # only avi and mp4 support
    vidfiles = glob(os.path.join(viddir, '*.mp4'))
    vidfiles = vidfiles + glob(os.path.join(viddir, '*.avi'))
    
    print(f'Found {len(vidfiles)} video files.')
    
    for vf in vidfiles:
        cap = cv2.VideoCapture(vf)

        # load the first frame
        success, frame = cap.read()
        # note that grayscale videos have 3 duplicate channels,
        frames = [frame[..., 0]]
        
        while success:
            success, frame = cap.read()
            if frame is not None:
                frames.append(frame[..., 0])

        video = np.stack(frames, axis=0)
        
        fdir = os.path.dirname(vf)
        fname = os.path.basename(vf)
        fext = fname.split('.')[-1]
        
        if 'video' not in fname.lower():
            suffix = '_video.nrrd'
        else:
            suffix = '.nrrd'
            
        outpath = os.path.join(fdir, fname.replace(fext, suffix))
        sitk.WriteImage(sitk.GetImageFromArray(video), outpath)
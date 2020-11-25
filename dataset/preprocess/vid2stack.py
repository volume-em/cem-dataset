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

    #read in the argument
    viddir = args.viddir
    
    #get a list of all mp4 and avi filepaths
    vidfiles = glob(os.path.join(viddir, '*.mp4'))
    vidfiles = vidfiles + glob(os.path.join(viddir, '*.avi'))
    
    print(f'Found {len(vidfiles)} video files.')
    
    for vf in vidfiles:
        #load the video into cv2
        cap = cv2.VideoCapture(vf)

        #load the first frame
        success, frame = cap.read()

        #loop over the video frames and store them in a list.
        #note that grayscale videos have 3 duplicate channels,
        #we only extract the first of these channels
        frames = [frame[:, :, 0]]
        while success:
            success, frame = cap.read()
            if frame is not None:
                frames.append(frame[:, :, 0])

        #stack the frames with the z-axis in the first
        #dimension
        video = np.stack(frames, axis=0)

        #save as nrrd files
        nrrd_path = '.'.join(vf.split('.')[:-1]) + '.nrrd'
        sitk.WriteImage(sitk.GetImageFromArray(video), nrrd_path)
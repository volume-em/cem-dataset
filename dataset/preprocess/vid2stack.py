import os, argparse
import cv2
import numpy as np
import SimpleITK as sitk
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vidpath', type=str, help='path to video file, avi or mp4')
    args = parser.parse_args()

    #open the video in cv2
    vidpath = args.vidpath
    
    vidfiles = glob(os.path.join(vidpath, '*.mp4'))
    vidfiles = vidfiles + glob(os.path.join(vidpath, '*.avi'))
    
    print(f'Found {len(vidfiles)} video files.')
    
    for vf in vidfiles:

        cap = cv2.VideoCapture(vf)

        #load the first frame (grayscale videos have 3 duplicate channels)
        success, frame = cap.read()

        #loop over the video frames and store them in a list
        frames = [frame[:, :, 0]]
        while success:
            success, frame = cap.read()
            if frame is not None:
                frames.append(frame[:, :, 0])

        #stack the frames
        video = np.stack(frames, axis=-1).transpose(2, 0, 1)

        #save as nrrd
        nrrd_path = '.'.join(vf.split('.')[:-1]) + '.nrrd'
        sitk.WriteImage(sitk.GetImageFromArray(video), nrrd_path)

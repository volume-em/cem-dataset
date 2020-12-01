"""
Description:
------------

It is assumed that this script will be run after the cross_section3d.py and 
crop_patches.py scripts. Errors are certain to occur if that is not the case.

This script takes a dask array of 2d image filepaths and a directory of 3d image 
volumes. It is assumed that at least some of the 2d images are cross sections from
the 3d volumes in the given directory. The cross_section3d.py creates a "trail" in
the 2d image filenames that make it easy to find their associated 3d volume.

Although it may seem circuitous to go from 3d volumes to 2d images back to 3d
volumes, this method allows us to use the simple and fast 2d filtering algorithms
to effectively filter out uninformative regions of a 3d volume. Reconstructions
are only performed in regions of the volume that contain a few "informative" 2d
patches.

Example usage:
--------------

python reconstruct3d.py {impaths_file} {volume_dir} {savedir} -nz 224 -p 4 --cross-plane

For help with arguments:
------------------------

python reconstruct3d.py --help
"""

import os, argparse, math
import numpy as np
import dask.array as da
import SimpleITK as sitk
from glob import glob
from multiprocessing import Pool

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('filtered_impaths_file', type=str, metavar='filtered_impaths_file', help='Dask array file with filtered impaths')
    parser.add_argument('volume_dir', type=str, metavar='volume_dir', help='Directory containing source EM volumes')
    parser.add_argument('savedir', type=str, metavar='savedir', 
                        help='Path to save 3d reconstructions')
    parser.add_argument('-nz', dest='nz', type=int, metavar='nz', default=224,
                        help='Number of z slices in reconstruction')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=32,
                        help='Number of processes to run, more processes run faster but consume memory')
    parser.add_argument('--cross-plane', dest='cross_plane', action='store_true',
                        help='Whether to create 3d volumes sliced orthogonal to imaging plane, useful when nz < image_shape')

    #parse the arguments
    args = parser.parse_args()
    filtered_impaths_file = args.filtered_impaths_file
    volume_dir = args.volume_dir
    savedir = args.savedir
    numberz = args.nz
    processes = args.processes
    cross_plane = args.cross_plane
    
    #to avoid running this long script only to get a nasty error
    #let's make sure that the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        
    #glob the volumes
    volume_paths = glob(os.path.join(volume_dir, '*'))
    print(f'Found {len(volume_paths)} in {volume_dir}')
    
    #extract the volume names
    #NOTE: this is the same code used to generate the names
    #from cross_section3d.py
    volume_names = []
    for vp in volume_paths:
        fext = vp.split('.')[-1] if vp[-5:] != 'nii.gz' else 'nii.gz'
        volume_names.append(vp.split('/')[-1].split(f'.{fext}')[0])
        
    volume_names = np.array(volume_names)
    
    #convert filtered to numpy straightaway
    #dask.array doesn't have good support for string operations
    filtered_impaths = da.from_npy_stack(filtered_impaths_file).compute()
    
    #the first thing that we need to do it to isolate
    #images from 3d source datasets. during creation we
    #gave 2d files the handy identifying -LOC-2d-
    source3d = np.where(np.core.defchararray.find(filtered_impaths, '-LOC-2d') == -1)
    print(f'Isolated {len(source3d[0])} images from 3d volumes out of {len(filtered_impaths)}')
    
    #overwrite filtered_impaths to save space
    #and sort the results such that images from the same
    #source datasets are grouped together
    filtered_impaths = np.sort(filtered_impaths[source3d])
    
    #just as in the deduplication script, we want to group
    #together images from the same source volume
    def get_dataset_name(imf):
        #function to extract the name of a dataset from the patch image file path
        #in the cross_section.py script we added the handy -LOC- indicator to
        #easily identify the source dataset from location information
        return imf.split('/')[-1].split('-LOC-')[0]

    #extract the set of unique dataset names from all the impaths
    with Pool(processes) as pool:
        datasets = np.sort(pool.map(get_dataset_name, filtered_impaths))

    #because we sorted the impaths, we know that all images from the
    #same dataset will be grouped together. therefore, we only need
    #to know the index of the first instance of a unique dataset name
    #in order to get the indices of all the patches from that dataset
    unq_datasets, indices = np.unique(datasets, return_index=True)
    
    #we can delete the datasets array
    del datasets
    
    #add the last index for impaths such that we have complete intervals
    #len(indices) == len(unq_datasets) + 1
    indices = np.append(indices, len(filtered_impaths))
    
    #get the intersect of unq_datasets and volume_names
    intersect_datasets, unq_indices, _ = np.intersect1d(unq_datasets, volume_names, return_indices=True)
    start_indices = indices[:-1][unq_indices]
    end_indices = indices[1:][unq_indices]

    #make groups of image patches by source dataset
    groups_impaths = []
    for si, ei in zip(start_indices, end_indices):
        #have to call .compute() for a dask array
        groups_impaths.append(filtered_impaths[si:ei])
        
    #we can delete the filtered_impaths and the indices
    del filtered_impaths, indices
    
    #define a function for non-maxmium suppression of
    #boxes in 3d
    def box_nms(boxes, scores, iou_threshold=0.2):
        #order the boxes by scores in descending
        #order (i.e. highest scores first)
        boxes = boxes[np.argsort(scores)[::-1]]
        
        #create a new list to save picked boxes
        picked_boxes = []
        
        #loop over boxes picking the highest scoring box
        #at each step and then eliminating any overlapping
        #boxes. continue until all the boxes have been exhausted
        while len(boxes) > 0:
            #pick the bounding box with largest confidence score
            #which will always be the first one in what's left of the
            #array (because we sorted boxes by score earlier)
            picked_boxes.append(boxes[0])
            
            #extract the coordinates from the boxes
            #(N, 6) --> 6 * (N, 1)
            z1, y1, x1, z2, y2, x2 = np.split(boxes, 6, axis=1)
        
            #calculate the volumes of all the remaining boxes
            #(N, 1) (by construction in this script, volumes will 
            #all be the same; this NMS function is generic though).
            volumes = (z2 - z1) * (y2 - y1) * (x2 - x1)

            #compute the intersections over union
            #between the first box and all boxes
            #IoU between the first box and itself will be 1
            #but we've already saved it
            zz1 = np.maximum(z1[0], z1)
            yy1 = np.maximum(y1[0], y1)
            xx1 = np.maximum(x1[0], x1)
            zz2 = np.minimum(z2[0], z2)
            yy2 = np.minimum(y2[0], y2)
            xx2 = np.minimum(x2[0], x2)

            #compute the volume of all the intersections
            d = np.maximum(0, zz2 - zz1)
            h = np.maximum(0, yy2 - yy1)
            w = np.maximum(0, xx2 - xx1)
            intersection_volumes = (d * h * w)

            #compute intersection over unions (N, 1)
            union_volumes = (volumes[0] + volumes - intersection_volumes)
            ious = intersection_volumes / union_volumes
            
            #indices of boxes to be removed
            remove_boxes = np.where(ious >= iou_threshold)[0]
            
            #update boxes (N, 6) --> (N-RB, 6)
            boxes = np.delete(boxes, remove_boxes, axis=0)
            
        return np.array(picked_boxes).astype('int')
    
    #alright now that all the setup is out of the way we want to
    #suppress images that would result in overlapping volumes.
    #by default we won't allow any overlap from stacks generated
    #from slices in the same plane, but we will allow up to 20%
    #overlap when slices come from different planes 
    #(20% percent is assuming using MoCo pretraining with 20%-100%
    #sized crops. This overlap criterion makes it less likely to have
    #identical cubes from two separate volumes.)
    def save_box_volumes(volume, dataset_name, boxes, is_isotropic):
        #convert to numpy
        volume = sitk.GetArrayFromImage(volume)
        
        if len(volume.shape) == 4:
            volume = volume[:, :, :, 0]
        
        #for filenames, get the digit padding
        zpad = math.ceil(math.log(volume.shape[0], 10))
        ypad = math.ceil(math.log(volume.shape[1], 10))
        xpad = math.ceil(math.log(volume.shape[2], 10))
        
        #the box indices are in the right format for
        #numpy dimensions because the each slices metadata
        #was constructed based on numpy dimensions (see cross_section3d.py)
        for box in boxes:
            z1, y1, x1, z2, y2, x2 = box
            zstr, ystr, xstr = str(z1).zfill(zpad), str(y1).zfill(ypad), str(x1).zfill(xpad)
            
            #extract the subvolume
            subvolume = volume[z1:z2, y1:y2, x1:x2]
                
            #if we're allowing cross plane, then we transpose
            #the dimensions such the
            if cross_plane:
                #order the dimensions from smallest to largest
                dim_order = np.argsort(subvolume.shape)
                dim_names = {0: 'z', 1: 'y', 2: 'x'}
                dim_str = ''.join([dim_names[d] for d in dim_order])
                subvolume = np.transpose(subvolume, tuple(dim_order))
            else:
                dim_str = 'zyx'
            
            #create slightly different file names if the dataset is
            #isotropic or not
            if is_isotropic:
                fname = f'{dataset_name}-LOC-3d-ISO-{zstr}_{ystr}_{xstr}_{dim_str}.npy'
            else:
                fname = f'{dataset_name}-LOC-3d-ANISO-{zstr}_{ystr}_{xstr}_{dim_str}.npy'
                
                
            #make sure that all dimensions are at least greater than
            #numberz // 2
            if all(s >= numberz // 2 for s in subvolume.shape):
                np.save(os.path.join(savedir, fname), subvolume)
        

    #first handle images from the same planes
    def overlap_suppression(impath_group):
        #at this point impath_group contains a bunch of paths
        #to images from th same dataset. in order to
        #extract areas in the 3d dataset to sample, we next want
        #to group them by "columns". a column is a group of images
        #that were sliced from the same plane and in the same
        #y and x location from within that plane. before we can
        #handle any of this we need to extract the metadata from the
        #filenames that appears after the -LOC- indicator
        #ex: {dataset_name}-LOC-{axis}_{slice_index}_{ys}_{xs}.tiff
        dataset_name = impath_group[0].split('/')[-1].split('-LOC-')[0]
        
        #if the dataset_name is not in volume names, then
        #we're going to skip it (this is in case the there are multiple
        #directories that containing source volumes that generated the images)
        #for example, we used 1 directory of volumes called 'internal' and
        #another called 'external'. meaning that this script needs to be
        #run twice (specifying the different volume_dir for each one)
        #find the index of the dataset_name
        #in volume_names
        try:
            vol_index = np.where(volume_names == dataset_name)[0][0]
            print(f'Found volume {dataset_name} at {vol_index}')
        except:
            print(f'Could not find volume {dataset_name}')
            return None
        
        #load the volume
        volume = sitk.ReadImage(volume_paths[vol_index])
        
        #get it's size in numpy style
        volsize = sitk.GetArrayFromImage(volume).shape
        
        axes = []
        slice_indices = []
        ys = []
        xs = []
        for f in impath_group:
            location_info = f.split('-LOC-')[-1].split('_')
            axes.append(int(location_info[0]))
            slice_indices.append(int(location_info[1]))
            ys.append(int(location_info[2]))
            xs.append(int(location_info[3].split('.tiff')[0]))
            
        axes = np.array(axes)
        slice_indices = np.array(slice_indices)
        ys = np.array(ys)
        xs = np.array(xs)
        
        #we're going to make a reasonable assumption that
        #if the set of images from a source volume only
        #has slices from axis 0 then that volume is likely
        #to be anisotropic. Ideally, this information would
        #be available from metadata on the volume; however,
        #this isn't alway the case. When we perform the cross
        #sectioning and filtering though, we tend to remove
        #images that have anisotropic pixels (the filtering nn
        #was trained to recognize these images as containing "artifacts")
        #so let's use this info: if unique axes are 0, then anisotropic
        #volume, otherwise, isotropic volume
        is_isotropic = True
        unique_axes = np.unique(axes)
        if len(unique_axes) == 1 and 0 in unique_axes:
            is_isotropic = False
        
        #there are a fewer nested loops that we need to run
        #through. the first are the plane axes: 0-yx, 1-zx, 2-zy
        boxes = []
        scores = []
        for axis in [0, 1, 2]:
            #get all the indices of images sliced from
            #the given axis (because of sorting they ought
            #to be contiguous)
            axis_indices = np.where(axes == axis)[0]
            if len(axis_indices) == 0:
                continue
            
            #extract unique pairs of y and x coordinates from
            #this axis
            axis_slice_indices = slice_indices[axis_indices]
            axis_ys = ys[axis_indices]
            axis_xs = xs[axis_indices]
            unique_2d = np.unique(np.stack([axis_ys, axis_xs], axis=1), axis=0)
            
            #the inner loop is to go through pairs of ys and xs
            #the so called "columns"
            for y,x in unique_2d:
                #extract the indices of the slices that
                #remained in the filtered impaths
                column_indices = np.where(np.logical_and(axis_ys == y, axis_xs == x))[0]
                column_slice_indices = axis_slice_indices[column_indices]

                #construct intervals of numberz thickness that run
                #from 0 to the largest slice index.
                intervals = np.arange(0, volsize[axis], numberz)

                #complete the intervals by appending the last index
                #from an additional interval that includes the maximum
                #slice index
                bins = np.append(intervals, volsize[axis])

                #we're about finished. last step is to append
                #the left edge of every interval that contains at least 1
                #one of the axis_slice_indices
                #the simplest way to do this is with a histogram
                counts, _ = np.histogram(column_slice_indices, bins=bins)
                #print(counts)
                
                #define the minimum number of informative
                #slices that must exist within an interval for
                #it to be considered informative
                min_count = numberz // 10

                #size is (n_good_intervals, 6)
                #(zstart, ystart, xstart, zend, yend, xend)
                column_boxes = np.zeros((len(intervals[counts > min_count]), 6))
                
                #size is (n_good_intervals, 6)
                #(zstart, ystart, xstart, zend, yend, xend)
                column_boxes = np.zeros((len(intervals[counts > min_count]), 6))
                column_boxes[:, axis] = intervals[counts > min_count]
                column_boxes[:, axis + 3] = intervals[counts > min_count] + numberz
                
                #of all the axes remove the one we're currently analyzing
                #e.g. testing axis 1: axes == [0, 1, 2] --> axes == [0, 2] 
                #--> y_axis == 0, x_axis == 2
                y_axis, x_axis = np.delete(np.arange(3), axis)
                
                #this assumes that the cropped image is 224x224
                #TODO: can this be adaptive without loading the image?
                column_boxes[:, y_axis] = y
                column_boxes[:, y_axis + 3] = y + 224
                column_boxes[:, x_axis] = x
                column_boxes[:, x_axis + 3] = x + 224
                
                scores.extend(counts[counts > min_count])
                boxes.extend(column_boxes)
                
        #convert boxes to an array
        boxes = np.array(boxes) #(N, 6)
        scores = np.array(scores) #(N,)
        
        #at this juncture the 3d boxes are such that cut boxes 
        #will have no overlap that is not transposed.
        #the last step is to remove overlaps that are transposed.
        #overlap is measured by the intersection-over-union of
        #two boxes. given a choice, we prefer boxes that contained
        #more informative images
        boxes = box_nms(boxes, scores, iou_threshold=0.1)
        
        #save results
        save_box_volumes(volume, dataset_name, boxes, is_isotropic)
        
        return None

    #get the sets of all boxes for reconstructing 3d data from
    #all the datasets
    with Pool(processes) as pool:
        result = list(pool.map(overlap_suppression, groups_impaths))

    print('Finished')
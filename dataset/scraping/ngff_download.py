"""
This script downloads image data from NGFF-style datasets stored either in s3 or locally.
Supported formats are N5, OME-ZARR, Neuroglancer precomputed.

The general download process is:

1) Open remote or local NGFF dataset with either fibsem_tools or CloudVolume.
2) Download the lowest resolution image data as an overview (reference) image.
3) Find up to X GBs of small ROIs in the overview image that are not mostly empty padding.
4) Crop and download ROIs from the dataset and save them as .nrrd images with correct voxel/pixel size.

Download and preprocessing instructions for a series of NGFF datasets are stored in a csv file.
That csv is the main argument to this script. ngff_datasets.csv in this repo provides a template.
The columns and formats are:

url: An optional url to the dataset description page.

source: Name of the database/source of an NGFF. Only required if the 
source is "openorganelle", otherwise this field is just a helpful descriptor.

api: The API to use for reading the dataset. Current options are "xarray", "CloudVolume",
and "ome_zarr". If uncertain, it's recommended to test all 3 manually.

download_url: Typically an s3 url for remote datasets and a filepath for locally stored datasets.

volume_name: The prefix name to give to all ROIs derived from that particular dataset. For safety
they should all be unique.

mip: The scale of data to download. 0 means full resoltion, 1 means half resolution, etc.

voxel_x: Voxel size in the x dimension. Only required for datasets that use the "xarray"
API. Voxel sizes can be read from file headers using the other APIs.

voxel_y: Voxel size in the y dimension. Only required for datasets that use the "xarray"
API. Voxel sizes can be read from file headers using the other APIs.

voxel_z: Voxel size in the z dimension. Only required for datasets that use the "xarray"
API. Voxel sizes can be read from file headers using the other APIs.

crop: Whether to crop the dataset into small ROIs or download it in toto. Should generally
be True for datasets larger than 5 GB.

crop_size: Cubic crop dimension. Usually 256.

invert: Whether to invert the pixel intensity of the downloaded ROI.

padding_value: The padding value that was used for the volume before any preprocessing.
0 for black padding and 255 for white padding.

Arguments:
----------
csv: The download csv file matching the format described above.

save_path: Directory in which to save the cropped ROIs.

--gb: Max amount of ROI data to crop from any dataset in gigabytes. Default 5.

Example Usage:
--------------

python scraping/ngff_download.py scraping/ngff_datasets.csv ./ngff_rois/ --gb 5

"""
import os, sys, math
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import io
from fibsem_tools import io as fibsem_io
import os, sys, math
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import io
from fibsem_tools import io as fibsem_io
from fibsem_tools.io import read_xarray
from cloudvolume import CloudVolume, Bbox
from tqdm import tqdm

def convert_to_byte(image):
    """
    Verify that image is byte type
    """
    if image.dtype == np.uint8:
        return image
    else:
        image = image.astype(np.float32)
        image -= image.min()
        im_max = image.max()
        if im_max > 0: # avoid zero division
            image /= im_max
            
        image *= 255
        
        return image.astype(np.uint8)
    
def load_volume(volume, invert=False):
    """
    Takes an xarray or CloudVolume and loads it
    as byte array. Optionally, inverts contrast.
    """
    volume = np.squeeze(np.array(volume[:]))
    volume = convert_to_byte(volume)
    if invert:
        volume = np.invert(volume)
        
    return volume

def sparse_roi_boxes(
    reference_volume,
    roi_size,
    padding_value=0,
    min_frac=0.7
):
    """
    Finds all ROIs of a given size within an overview volume
    that contain at least some non-padding values.
    
    Arguments:
    ----------
    reference_volume (np.ndarray): A low-resolution overview image
    that can fit in memory. Typically this is the lowest-resolution
    available in an image pyramid.
    
    roi_size (Tuple[d, h, w]): Size of ROIs in voxels relative to 
    the reference volume.
    
    padding_value (float): Value used to pad the reference volume.
    Must be confirmed by manual inspection.
    
    min_frac (float): Fraction from 0-1 of and ROI that must be
    non-padding values.
    
    Returns:
    --------
    roi_boxes (np.ndarray): Array of (N, 6) defining bounding boxes
    for ROIs that passed that min_frac condition.
    
    """
    # grid for box indices
    xcs, ycs, zcs = roi_size
    xsize, ysize, zsize = reference_volume.shape
    
    xgrid = np.arange(0, xsize + 1, xcs)
    ygrid = np.arange(0, ysize + 1, ycs)
    zgrid = np.arange(0, zsize + 1, zcs)
    
    max_padding = (1 - min_frac) * np.prod(roi_size)

    # make sure that there's always an ending index
    # so we have complete ranges
    if len(xgrid) < 2 or xsize % xcs > 0.5 * xcs:
        xgrid = np.append(xgrid, np.array(xsize)[None], axis=0)
    if len(ygrid) < 2 or ysize % ycs > 0.5 * ycs:
        ygrid = np.append(ygrid, np.array(ysize)[None], axis=0)
    if len(zgrid) < 2 or zsize % ycs > 0.5 * zcs:
        zgrid = np.append(zgrid, np.array(zsize)[None], axis=0)

    roi_boxes = []
    for xi, xf in zip(xgrid[:-1], xgrid[1:]):
        for yi, yf in zip(ygrid[:-1], ygrid[1:]):
            for zi, zf in zip(zgrid[:-1], zgrid[1:]):
                box_slices = tuple([slice(xi, xf), slice(yi, yf), slice(zi, zf)])
                n_not_padding = np.count_nonzero(
                    reference_volume[box_slices] == padding_value
                )
                if n_not_padding < max_padding:
                    roi_boxes.append([xi, yi, zi, xf, yf, zf])
                    
    return np.array(roi_boxes)

def crop_cloud_volume(
    url, 
    save_path,
    volume_name, 
    target_mip, 
    n_cubes=100,
    cube_size=256,
    invert=False,
    padding_value=0,
    min_frac=0.7
):
    """
    Crops non-empty ROIs from a CloudVolume and saves
    the results as .nrrd images with correct voxel size.
    
    Arguments:
    ----------
    url (str): URL from which to load the CloudVolume.
    
    save_path (str): Directory in which to save crops.
    
    volume_name (str): Name used to identify this volume. It
    will be the prefix of all crop filenames.
    
    target_mip (int): The mip level from which to crop data.
    
    n_cubes (int): The maximum number of subvolumes to crop
    from the CloudVolume.
    
    cube_size (int): The size of crops. Assumes crops are have
    cubic dimensions.
    
    invert (bool): Whether to invert the intensity of the image.
    
    padding_value (float): Value in the CloudVolume used as image
    padding. If invert is True, this value will be inverted as well.
    
    min_frac (float): Fraction from 0-1 of and ROI that must be
    non-padding values.
    
    """
    # check available mip levels
    high_mip = target_mip
    volume_high = CloudVolume(url, mip=high_mip, use_https=True, 
                              fill_missing=True, progress=False)
    
    cv_box = np.array(volume_high.bounds.to_list())
    x0, y0, z0 = cv_box[:3]
    
    if invert:
        padding_value = 255 - padding_value
    
    # get the nm resolution
    high_resolution = list(volume_high.available_resolutions)[high_mip]

    # use lowest resolution as reference, unless
    # it's a factor of cube_size smaller than the scale
    # at the high mip level
    low_mip = max(list(volume_high.available_mips))
    factor = (2 ** (low_mip + 1)) / (2 ** (high_mip + 1))
    if factor >= cube_size:
        max_mip_diff = int(math.log(cube_size, 2)) - 1
        low_mip = high_mip + max_mip_diff

    volume_low = CloudVolume(url, mip=low_mip, use_https=True, 
                             fill_missing=True, progress=True)
    
    # check whether the reference volume already exists
    # create it if not
    reference_fpath = os.path.join(save_path, f'{volume_name}_reference_mip{low_mip}.tif')
    if not os.path.exists(reference_fpath):
        reference_volume = load_volume(volume_low, invert)
        io.imsave(reference_fpath, reference_volume)
    
    reference_volume = io.imread(reference_fpath)

    # find possible ROIs that are not just blank padding 
    factors = np.array(volume_high.shape[:3]) / np.array(volume_low.shape[:3])
    low_res_roi_size = np.floor(cube_size / factors).astype('int')
    roi_boxes = sparse_roi_boxes(
        reference_volume, low_res_roi_size, padding_value, min_frac
    )

    # randomly select n_cubes ROI boxes
    box_indices = np.random.choice(
        range(len(roi_boxes)), size=(min(n_cubes, len(roi_boxes)),), replace=False
    )
    bboxes_low = roi_boxes[box_indices]

    # loop through the boxes that we selected
    for bbox_low in tqdm(bboxes_low):
        # convert between mips
        bbox_low = Bbox(bbox_low[:3], bbox_low[3:])

        # convert the ROI box from low to high resolution scale
        bbox_high = np.array(
            volume_high.bbox_to_mip(bbox_low, low_mip, high_mip).to_list()
        )

        # add bounding indices as an offset
        bbox_high[0] += x0
        bbox_high[3] += x0
        bbox_high[1] += y0
        bbox_high[4] += y0
        bbox_high[2] += z0
        bbox_high[5] += z0
        
        # boundaries aren't always consistent
        # clip the bbox by the volume bounds
        bbox_high1 = np.clip(bbox_high[:3], cv_box[:3], cv_box[3:])
        bbox_high2 = np.clip(bbox_high[3:], cv_box[:3], cv_box[3:])
        
        bbox_high = np.concatenate([bbox_high1, bbox_high2])

        bbox_high_slices = tuple([
            slice(bbox_high[0], bbox_high[3]),
            slice(bbox_high[1], bbox_high[4]),
            slice(bbox_high[2], bbox_high[5])
        ])

        # handle case of an unitary channel
        if len(volume_high.shape) == 4:
            bbox_high_slices += (0,)

        # crop the high-resolution cube
        cube = load_volume(volume_high[bbox_high_slices], invert)

        # extract ranges for filename
        x1, x2 = bbox_high[0], bbox_high[3]
        y1, y2 = bbox_high[1], bbox_high[4]
        z1, z2 = bbox_high[2], bbox_high[5]

        cube_fname = f'{volume_name}_s{mip_high}-ROI-x{x1}-{x2}_y{y1}-{y2}_z{z1}-{z2}.nrrd'
        cube_fpath = os.path.join(save_path, cube_fname)
        
        # transpose from xyz to zyx
        cube = cube.transpose(2, 1, 0)
        
        # save as nrrd with appropriate voxel size in header
        cube = sitk.GetImageFromArray(cube)
        cube.SetSpacing(high_resolution)
        sitk.WriteImage(cube, cube_fpath)

def crop_ome_zarr(
    url, 
    save_path,
    volume_name,
    source,
    target_mip,
    n_cubes=100,
    cube_size=256,
    invert=False,
    padding_value=0,
    min_frac=0.7
):
    """
    Crops non-empty ROIs from a xarray and saves
    the results as .nrrd images with correct voxel size.
    
    Arguments:
    ----------
    url (str): URL from which to load the xarray.
    
    save_path (str): Directory in which to save crops.
    
    volume_name (str): Name used to identify this volume. It
    will be the prefix of all crop filenames.
    
    target_mip (int): The mip level from which to crop data.
    
    n_cubes (int): The maximum number of subvolumes to crop
    from the CloudVolume.
    
    cube_size (int): The size of crops. Assumes crops are have
    cubic dimensions.
    
    invert (bool): Whether to invert the intensity of the image.
    
    padding_value (float): Value in the CloudVolume used as image
    padding. If invert is True, this value will be inverted as well.
    
    min_frac (float): Fraction from 0-1 of and ROI that must be
    non-padding values.
    
    """
    # url to the dataset we want to download
    high_mip = target_mip
    mip_str = f'{target_mip}'
    
    data = fibsem_io.read(url)
    volume_high = data[mip_str]
    res_metadata = data.attrs['multiscales'][0]['datasets'][int(mip_str)]['coordinateTransformations']
    high_resolution = res_metadata[0]['scale']
        
    # only way to check available mip levels
    # is 1 by 1 start from the smallest desirable
    max_mip_diff = int(math.log(cube_size, 2)) - 1
    low_mip = high_mip + max_mip_diff
    
    for mip in list(range(low_mip, high_mip - 1, -1)):
        mip_str = f'{mip}'
        try:
            volume_low = data[mip_str]
            low_mip = mip
            break
        except Exception as err:
            continue
            
    if low_mip <= high_mip:
        raise Exception('No low resolution volumes found! Are you sure?')

    # check whether the reference volume already exists
    # create it if not
    reference_fpath = os.path.join(save_path, f'{volume_name}_reference_mip{low_mip}.tif')
    if not os.path.exists(reference_fpath):
        reference_volume = load_volume(volume_low, invert)
        io.imsave(reference_fpath, reference_volume)
    
    reference_volume = io.imread(reference_fpath)
            
    # find possible ROIs that are not just blank padding 
    # they may still be uniform resin though (filtering
    # will happen later)
    factors = np.array(volume_high.shape[:3]) / np.array(volume_low.shape[:3])
    low_res_roi_size = np.floor(cube_size / factors).astype('int')
    roi_boxes = sparse_roi_boxes(
        reference_volume, low_res_roi_size, padding_value, min_frac
    )
    
    # randomly select n_cubes ROI boxes
    box_indices = np.random.choice(
        range(len(roi_boxes)), size=(min(n_cubes, len(roi_boxes)),), replace=False
    )
    bboxes_low = roi_boxes[box_indices]

    # loop through the boxes that we selected
    for bbox_low in tqdm(bboxes_low):
        # convert between mips
        bbox_high = np.concatenate(
            [bbox_low[:3] * factors, bbox_low[3:] * factors]
        ).astype('int')
        
        # boundaries aren't always consistent
        # clip the bbox by the volume size
        bbox_high1 = np.clip(bbox_high[:3], 0, None)
        bbox_high2 = np.clip(bbox_high[3:], None, volume_high.shape[:3])
        bbox_high = np.concatenate([bbox_high1, bbox_high2])
        
        # make sure it's actually a volume with 5 slices
        # skip this bounding box otherwise (it's on the edge)
        if np.any(bbox_high[3:] - bbox_high[:3] < 5):
            continue

        bbox_high_slices = tuple([
            slice(bbox_high[0], bbox_high[3]),
            slice(bbox_high[1], bbox_high[4]),
            slice(bbox_high[2], bbox_high[5])
        ])

        # handle case of a unitary channel
        if volume_high.ndim == 4:
            bbox_high_slices += (0,)

        # crop the cube
        cube = load_volume(volume_high[bbox_high_slices], invert)

        x1, x2 = bbox_high[0], bbox_high[3]
        y1, y2 = bbox_high[1], bbox_high[4]
        z1, z2 = bbox_high[2], bbox_high[5]
        cube_fname = f'{volume_name}_s{high_mip}-ROI-x{x1}-{x2}_y{y1}-{y2}_z{z1}-{z2}.nrrd'
        cube_fpath = os.path.join(save_path, cube_fname)
        
        cube = sitk.GetImageFromArray(cube)
        cube.SetSpacing(high_resolution)
        sitk.WriteImage(cube, cube_fpath)
        
def crop_xarray(
    url, 
    save_path,
    volume_name,
    source,
    target_mip,
    high_resolution,
    n_cubes=100,
    cube_size=256,
    invert=False,
    padding_value=0,
    min_frac=0.7,
    storage_options=None
):
    """
    Crops non-empty ROIs from a xarray and saves
    the results as .nrrd images with correct voxel size.
    
    Arguments:
    ----------
    url (str): URL from which to load the xarray.
    
    save_path (str): Directory in which to save crops.
    
    volume_name (str): Name used to identify this volume. It
    will be the prefix of all crop filenames.
    
    target_mip (int): The mip level from which to crop data.

    high_resolution (Tuple[float]): Resolution in nanometers
    of the target_mip data. Should be z y x size respectively.
    
    n_cubes (int): The maximum number of subvolumes to crop
    from the CloudVolume.
    
    cube_size (int): The size of crops. Assumes crops are have
    cubic dimensions.
    
    invert (bool): Whether to invert the intensity of the image.
    
    padding_value (float): Value in the CloudVolume used as image
    padding. If invert is True, this value will be inverted as well.
    
    min_frac (float): Fraction from 0-1 of and ROI that must be
    non-padding values.
    
    storage_options (Dict): Storage options for reading an xarray.
    
    """
    # url to the dataset we want to download
    high_mip = target_mip
    mip_str = f's{target_mip}'
    mip_url = os.path.join(url, mip_str)
    
    volume_high = read_xarray(mip_url, storage_options=storage_options)
        
    # only way to check available mip levels
    # is 1 by 1 start from the smallest desirable
    max_mip_diff = int(math.log(cube_size, 2)) - 1
    low_mip = high_mip + max_mip_diff
    
    for mip in list(range(low_mip, high_mip - 1, -1)):
        mip_str = f's{mip}'
        mip_url = os.path.join(url, mip_str)
        try:
            volume_low = read_xarray(mip_url, storage_options=storage_options)
            low_mip = mip
            break
        except Exception as err:
            continue
            
    if low_mip <= high_mip:
        raise Exception('No low resolution volumes found! Are you sure?')

    # check whether the reference volume already exists
    # create it if not
    reference_fpath = os.path.join(save_path, f'{volume_name}_reference_mip{low_mip}.tif')
    if not os.path.exists(reference_fpath):
        reference_volume = load_volume(volume_low, invert)
        io.imsave(reference_fpath, reference_volume)
    
    reference_volume = io.imread(reference_fpath)
            
    # find possible ROIs that are not just blank padding 
    # they may still be uniform resin though (filtering
    # will happen later)
    factors = np.array(volume_high.shape[:3]) / np.array(volume_low.shape[:3])
    low_res_roi_size = np.floor(cube_size / factors).astype('int')
    roi_boxes = sparse_roi_boxes(
        reference_volume, low_res_roi_size, padding_value, min_frac
    )
    
    # randomly select n_cubes ROI boxes
    box_indices = np.random.choice(
        range(len(roi_boxes)), size=(min(n_cubes, len(roi_boxes)),), replace=False
    )
    bboxes_low = roi_boxes[box_indices]

    # loop through the boxes that we selected
    for bbox_low in tqdm(bboxes_low):
        # convert between mips
        bbox_high = np.concatenate(
            [bbox_low[:3] * factors, bbox_low[3:] * factors]
        ).astype('int')
        
        # boundaries aren't always consistent
        # clip the bbox by the volume size
        bbox_high1 = np.clip(bbox_high[:3], 0, None)
        bbox_high2 = np.clip(bbox_high[3:], None, volume_high.shape[:3])
        bbox_high = np.concatenate([bbox_high1, bbox_high2])
        
        # make sure it's actually a volume with 5 slices
        # skip this bounding box otherwise (it's on the edge)
        if np.any(bbox_high[3:] - bbox_high[:3] < 5):
            continue

        bbox_high_slices = tuple([
            slice(bbox_high[0], bbox_high[3]),
            slice(bbox_high[1], bbox_high[4]),
            slice(bbox_high[2], bbox_high[5])
        ])

        # handle case of a unitary channel
        if volume_high.ndim == 4:
            bbox_high_slices += (0,)

        # crop the cube
        cube = load_volume(volume_high[bbox_high_slices], invert)

        x1, x2 = bbox_high[0], bbox_high[3]
        y1, y2 = bbox_high[1], bbox_high[4]
        z1, z2 = bbox_high[2], bbox_high[5]
        cube_fname = f'{volume_name}_s{high_mip}-ROI-x{x1}-{x2}_y{y1}-{y2}_z{z1}-{z2}.nrrd'
        cube_fpath = os.path.join(save_path, cube_fname)
        
        cube = sitk.GetImageFromArray(cube)
        cube.SetSpacing(high_resolution)
        sitk.WriteImage(cube, cube_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='path to url csv file')
    parser.add_argument('save_path', type=str, help='path to save the volumes')
    parser.add_argument('--gb', type=float, dest='max_gbs', default=5, help='maximum number of GBs to crop')
    args = parser.parse_args()

    # load the csv
    df = pd.read_csv(args.csv)
    save_path = args.save_path
    max_gbs = args.max_gbs
    
    os.makedirs(save_path, exist_ok=True)

    for i, row in df.iterrows():
        source = row['source']
        url = row['download_url']
        volname = row['volume_name']
        mip = row['mip']
        crop = row['crop']
        crop_size = row['crop_size']
        invert = row['invert']
        padding_value = row['padding_value']
        api = row['api']
        
        # read resolution from csv
        # CloudVolume api reads it directly from metadata
        if api in ['xarray']:
            xyz = [
                float(row['voxel_x']), float(row['voxel_y']), float(row['voxel_z'])
            ]
            resolution = (mip + 1) * np.array(xyz)
            resolution = resolution.tolist()
            if source in ['openorganelle']:
                storage_options = {'anon' : True}
            else:
                storage_options = None

        # compute the number of cubes from the crop_size
        bytes_per_cube = crop_size ** 3
        n_cubes = int(max(1, (max_gbs * 1024 ** 3) // bytes_per_cube))
        
        print(f'Downloading from {url}')
        if crop and api in ['CloudVolume']:
            crop_cloud_volume(
                url, 
                save_path, 
                volname, 
                mip, 
                n_cubes, 
                crop_size, 
                invert, 
                padding_value
            )
        elif crop and api in ['xarray']:
            crop_xarray(
                url,
                save_path,
                volname,
                source,
                mip,
                resolution,
                n_cubes,
                crop_size,
                invert,
                padding_value,
                storage_options=storage_options
            )
        elif crop and api in ['ome_zarr']:
            crop_ome_zarr(
                url,
                save_path,
                volname,
                source,
                mip,
                n_cubes,
                crop_size,
                invert,
                padding_value
            )
        elif not crop and api in ['CloudVolume']:
            volume = CloudVolume(url, mip=mip, use_https=True, 
                                 fill_missing=True, progress=False)
            
            # get the nm resolution
            resolution = list(volume.available_resolutions)[mip]
            
            # download and process the volume
            volume = load_volume(volume, invert)
            
            # transpose from xyz to zyx
            volume = volume.transpose(2, 1, 0)
            vol_fpath = os.path.join(save_path, f'{volname}_s{mip}.nrrd')

            volume = sitk.GetImageFromArray(volume)
            volume.SetSpacing(resolution)
            sitk.WriteImage(volume, vol_fpath)
        elif not crop and api in ['xarray']:
            mip_str = f's{mip}'
            mip_url = os.path.join(url, mip_str)

            volume = read_xarray(mip_url, storage_options=storage_options)
                
            # download and process the volume
            volume = load_volume(volume, invert)
            vol_fpath = os.path.join(save_path, f'{volname}_{mip_str}.nrrd')

            volume = sitk.GetImageFromArray(volume)
            volume.SetSpacing(resolution)
            sitk.WriteImage(volume, vol_fpath)
        elif not crop and api in ['ome_zarr']:
            mip_str = f'{mip}'
            data = fibsem_io.read(url)
            res_metadata = data.attrs['multiscales'][0]['datasets'][int(mip_str)]['coordinateTransformations']
            resolution = res_metadata[0]['scale']
            volume = data[mip_str]
                
            # download and process the volume
            volume = load_volume(volume, invert)
            vol_fpath = os.path.join(save_path, f'{volname}_s{mip_str}.nrrd')

            volume = sitk.GetImageFromArray(volume)
            volume.SetSpacing(resolution)
            sitk.WriteImage(volume, vol_fpath)
        else:
            raise Exception('Nothing to do with this dataset!')

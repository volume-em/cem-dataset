"""
Download script for the H01 Dataset
See: https://h01-release.storage.googleapis.com/landing.html

"""

import os
import numpy as np
import SimpleITK as sitk
# Ensure tensorstore does not attempt to use GCE credentials
os.environ['GCE_METADATA_ROOT'] = 'metadata.google.internal.invalid'
import tensorstore as ts


context = ts.Context({'cache_pool': {'total_bytes_limit': 1000000000}})
volname = 'shapson-coe2021_h01'

em_8nm = ts.open({
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'h01-release'},
    'path': 'data/20210601/8nm_raw'},
    read=True, context=context).result()[ts.d['channel'][0]]

xsize, ysize, zsize = em_8nm.shape

# best option is to randomly pick cubes
# of given size
crop_size = 512
max_gbs = 5

# compute the number of cubes from the crop_size
bytes_per_cube = crop_size ** 3
n_cubes = int((max_gbs * 1024 ** 3) / bytes_per_cube)

save_path = './H01'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# crop and save many will be blank padding
n_cropped = 0
while n_cropped < n_cubes:
    # pick indices for n_cubes
    # ranges set manually from checking
    # overview in neuroglancer
    x = np.random.randint(70000, 450000)
    y = np.random.randint(40000, 300000)
    z = np.random.randint(0, zsize - crop_size)
    xf = x + crop_size
    yf = y + crop_size
    zf = z + crop_size
    
    # check if first pixel is
    # zero, if yes then skip
    if em_8nm[x, y, z].read().result() == 0:
        continue
    else:
        cube = em_8nm[x:xf, y:yf, z:zf].read().result()

        # this dataset is already uint8 and inverted
        # we only need to transpose from xyz to zyx
        cube = cube.transpose(2, 1, 0)

        # save the result with given resolution
        cube_fname = f'{volname}-ROI-x{x}-{xf}_y{y}-{yf}_z{z}-{zf}.nrrd'
        cube_fpath = os.path.join(save_path, cube_fname)

        cube = sitk.GetImageFromArray(cube)
        cube.SetSpacing([8, 8, 30])
        sitk.WriteImage(cube, cube_fpath)
        
        n_cropped += 1
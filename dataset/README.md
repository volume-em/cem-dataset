# Dataset Curation

The scripts in this directory handle all dataset preprocessing and curation. Below is an example workflow, read the script headers for more details. Or to see all available parameters use:

```bash
python {script_name}.py --help
```

**Note: The ```patchify2d.py```, ```patchify3d.py```,  and ```classify_patches.py``` scripts are all designed for continuous integration. Datasets that have been processed previously and are in the designated output directories will be ignored by all of them.**

## 2D Data Preparation

2D images are expected to be organized into directories, where each directory contains a group of images generated
as part of the same imaging project or at least with roughly the same biological metadata.

First, standardize images to single channel grayscale and unsigned 8-bit:

```bash
# make copies in new_directory
python preprocess/cleanup2d.py {dir_of_2d_image_groups} -o {new_directory} --processes 4
# or, instead, overwrite images inplace
python preprocess/cleanup2d.py {dir_of_2d_image_groups} --processes 4
```
Second, crop each image into fixed size patches (typically 224x224):

```bash
python patchify2d.py {dir_of_2d_image_groups} {patch_dir} -cs 224 --processes 4
```

The ```patchify2d.py``` script will save a ```.pkl``` file with the name of each 2D image subdirectory. Pickle files contain a dictionary of patches from all images in the subdirectory along with corresponding filenames. These files are ready for filtering (see below).

## Video Preparation

Convert videos in ```.avi``` or ```.mp4``` format to ```.nrrd``` images with correct naming convention (i.e., put the word 'video' in the filename).

```bash
python preprocess/vid2stack.py {dir_of_videos}
```

## 3D Data Preparation

3D datasets are expected to be in a single directory (this includes any video stacks created in the previous section). Supported formats are anything that can be [read by SimpleITK](https://simpleitk.readthedocs.io/en/v1.2.3/Documentation/docs/source/IO.html). It's important that if any volumes are in ```.mrc``` format they be converted to unsigned bytes. With IMOD installed this can be done using:

```bash
python preprocess/mrc2byte.py {dir_of_mrc_files}
```

Next, cross-section, patch, and deduplicate volume files. If processing a combination of isotropic and anisotropic volumes, it's crucial that each dataset has a correct header recording the voxel size. If Z resolution is greater that 25% 
different from xy resolution, then cross-sections will only be cut from the xy plane, even if axes 0, 1, 2 are passed to the script (see usage example below). 

```bash
python patchify3d.py {dir_of_3d_datasets} {patch_dir} -cs 224 --axes 0 1 2 --processes 4
```

The ```patchify3d.py``` script will save a ```.pkl``` file with the name of each volume file. Pickle files contain a dictionary of patches along with corresponding filenames. These files are ready for filtering (see below).

## Filtering 

2D, video, and 3D datasets can be filtered with the same script just put all the ```.pkl``` files in the same directory. By default, filtering uses a ResNet34 model that was trained on 12,000 manually annotated patches. The weights for this model are downloaded from [Zenodo](https://zenodo.org/record/6458015#.YlmNaS-cbTR) automatically. A new model can be trained, if needed, using the ```train_patch_classifier.py``` script.

Filtering will be fastest with a GPU installed, but it's not required.

```bash
python classify_patches.py {patch_dir} {save_dir}
```

After running filtering, the ```save_dir``` with have one subdirectory for each of the ```.pkl``` files that were processed. Each subdirectory contains single channel grayscale, unsigned 8-bit tiff images.



# Dataset Curation

Beginning with a collection of 2d and 3d EM images, the scripts in this directory handle all of the dataset preprocessing and curation.

For preliminary data preparation the vid2stack.py and mrc2byte.py scripts in the preprocess directory convert videos into image volumes and mrc volumes from signed to unsigned bytes.

The main curation pipeline starts with the cross-sectioning of 3d volumes into 2d image slices that can be combined together with any 2d EM datasets. Cross-sectioning for 3d data is handled by the raw/cross_section3d.py script and some basic type checking and file renaming for 2d data is done by the raw/cleanup2d.py script. It's recommended that 2d and 3d "corpuses" of data be kept in separate directories in order to ensure that the two scripts run smoothly; however, the outputs from the raw/cleanup2d.py and raw/cross_section3d.py scripts should all be saved in the same directory. The collection of 2d that results, which are all 8-bit unsigned tiffs, can then be cropped into patches of a given size using the raw/crop_patches.py script. In summary the first step of the workflow is:

1. Run raw/cleanup2d.py on directory of 2d EM images. Save results to *save_dir*
2. Run raw/cross_section3d.py on directory of 3d EM images. Save results to *save_dir*
3. Run raw/rop_patches.py on images in *save_dir*. Save results to *raw_save_dir*

The completion of this first step in the workflow yields the *Raw* dataset. Note that the raw/crop_patches.py sript not only creates tiff images for each of the patches, but also creates a numpy array of the patch's difference hash. The hashes are used for deduplication.

Deduplication uses the deduplicated/deduplicate.py script. As input the script expects *raw_save_dir* containing the .tiff images and .npy hashes. If new data is added to the *raw_save_dir* after the deduplication script has already been run, the script will only deduplicate the new datasets. This makes it easy to add new datasets without the somewhat time-consuming burden of rerunning deduplication for the entire *Raw* dataset. In summary:

1. Run deduplicated/deduplicate.py on *raw_save_dir*. Save results, which are .npy files for each 2d/3d dataset that contain a list of filepaths for exemplar images, to *deduplicated_save_dir*.

The addition to .npy files for each datasets, the script also outputs a dask array file called deduplicated_fpaths.npz that contains the list of file paths for exemplar images from all 2d/3d datasets. This collection of file paths defines the *Deduplicated* dataset.

In the last curation step, uninformative patches are filtered out using a ResNet34 classifier. The filtered/train_nn.py script trains the classifier on a collection of manually labeled image files contained in deduplicated_fpaths.npz. It is assumed that the labeling was performed using the labeling.ipynb notebook included in this repository. In general, training a new classifier shouldn't be necessary; we release the weights for the classifer that we trained on 12,000 labeled images. The filtered/classify_nn.py script performs inference on the set of unlabeled images in deduplicated_fpaths.npz. By default, the script will download and use the weights that we released. In summary:

1. (Optional) Manually label images in deduplicated_fpaths.npz using labeling.ipynb.
2. (Optional) Run filtered/train_nn.py to train and evaluate a ResNet34 on the images labeled in step 1.
3. Run filtered/classify.py on images images in deduplicated_fpaths.npz. Save dask array of all informative images, nn_filtered_fpaths.npz, to *filtered_save_dir*.

These last steps result in the *Filtered* dataset. That's the complete curation pipeline. An optional last step, to generate 3d data, is to run the 3d/reconstruct3d.py script. This script takes the set of filtered images, nn_filtered_fpaths.npz, and the original directory of 3d volumes (i.e. the directory given to cross_section3d.py earlier) and makes data volumes of a given z-thickness. Note that one limitation of this script is that it currently assumes patches are 224x224.




# CellEMNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cem500k-a-large-scale-heterogeneous-unlabeled/electron-microscopy-image-segmentation-on-1)](https://paperswithcode.com/sota/electron-microscopy-image-segmentation-on-1?p=cem500k-a-large-scale-heterogeneous-unlabeled)


Code for the paper: [CEM500K - A large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning.](https://www.biorxiv.org/content/10.1101/2020.12.11.421792v1)


## Getting Started

First clone this repository:

```
git clone https://github.com/volume-em/cellemnet
```

If using conda, install dependencies in a new environment:

```
cd cellemnet
conda env create -f environment.yml
```

Otherwise, required dependencies can be installed with another package manager (pip):
- torch
- torchvision
- segmentation-models-pytorch
- albumentations
- h5py
- mlflow
- simpleitk
- scikit-learn
- imagehash

## Download CEM500K

The CEM500K dataset, metadata and pretrained_weights are available through [EMPIAR ID 10592](https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10592/).

## Use the pre-trained weights

Currently, pre-trained weights are only available for PyTorch. For an example of how to use them see ```evaluation/benchmark_configs``` and ```notebooks/pretrained_weights.ipynb```.

We're working to convert the weights for use with TensorFlow/Keras. If you have any experience with this kind of conversion and would like to help with testing, please open an issue.

## Data Curation

For image deduplication and filtering routines see the ```dataset``` directory README. Results on a small example 3D image volume can be reviewed in ```notebooks/deduplication_and_filtering.ipynb```.

## Citing this work

Please cite this work.
```
@article {Conrad2020.12.11.421792,
	author = {Conrad, Ryan W and Narayan, Kedar},
	title = {CEM500K - A large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning.},
	elocation-id = {2020.12.11.421792},
	year = {2020},
	doi = {10.1101/2020.12.11.421792},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/12/11/2020.12.11.421792},
	eprint = {https://www.biorxiv.org/content/early/2020/12/11/2020.12.11.421792.full.pdf},
	journal = {bioRxiv}
}
```
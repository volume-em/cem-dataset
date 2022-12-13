# CellEMNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cem500k-a-large-scale-heterogeneous-unlabeled/electron-microscopy-image-segmentation-on-1)](https://paperswithcode.com/sota/electron-microscopy-image-segmentation-on-1?p=cem500k-a-large-scale-heterogeneous-unlabeled)

Code for the paper: [CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning](https://elifesciences.org/articles/65894)


## Getting Started

First clone this repository:

```
git clone https://github.com/volume-em/cem-dataset.git
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

## Download the Dataset

The latest iteration of the CEM dataset is CEM1.5M. Images and metadata are available for download through [EMPIAR ID 11035](https://www.ebi.ac.uk/empiar/EMPIAR-11035/).

## Pre-trained weights

Currently, pre-trained weights are only available for PyTorch. For an example of how to use them see ```evaluation/benchmark_configs``` and ```notebooks/pretrained_weights.ipynb```.

| Model architecture  | Pre-training method | Dataset     | Link                                           |
| ------------------- | ------------------- | ----------- | ---------------------------------------------- |
| ResNet50            | MoCoV2              | CEM500K     | https://zenodo.org/record/6453140#.Y5inAC2B1Qg |
| ResNet50            | SWaV                | CEM1.5M     | https://zenodo.org/record/6453160#.Y5iznS2B1Qh |



## Data Curation

For image deduplication and filtering routines see the ```dataset``` directory README. Results on a small example 3D image volume can be reviewed in ```notebooks/deduplication_and_filtering.ipynb```.

## Citing this work

Please cite this work.

```bibtex
@article {Conrad2021,
	author = {Conrad, Ryan and Narayan, Kedar},
	doi = {10.7554/eLife.65894},
	issn = {2050-084X},
	journal = {eLife},
	month = {apr},
	title = {{CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning}},
	url = {https://elifesciences.org/articles/65894},
	volume = {10},
	year = {2021}
}
```
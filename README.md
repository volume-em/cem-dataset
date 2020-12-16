# CellEMNet

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

<ul>
    <li>torch</li>
    <li>torchvision</li>
    <li>segmentation-models-pytorch</li>
    <li>albumentations</li>
    <li>h5py</li>
    <li>mlflow</li>
    <li>simpleitk</li>
    <li>scikit-learn</li>
    <li>imagehash</li>
</ul>

## Download pre-trained weights

Currently, pre-trained weights are only available for PyTorch. For an example of how to use them see ```evaluation/benchmark_configs```. Configuration files provide the option to use 'cellemnet_mocov2' as the pre-training.

We're working to convert the weights for use with TensorFlow/Keras. If you have any experience with this kind of conversion and would like to help with testing, please open an issue.

## Download CEM500K

The dataset is currently awaiting deposition on EMPIAR. Updates on the ID and download instructions coming soon.

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

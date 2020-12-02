# CellEMNet

Code for the paper: LINK HERE ONCE THE PAPER IS PUBLISHED

NOTE: For now these are aspirational until we get the hosting figured out. 

## Download the Dataset

The dataset is currently hosted on ?????. To download:

```
wget https://somewebsite.com/{dataset_id}
```

Dataset IDs for the versions of the CellEMNet dataset are:

| Dataset name  | Image sizes   | Dataset ID       | Number of images | Date Updated  | 
| ------------- | ------------- | ---------------- | ---------------- | ------------- |
| CEM500K       | 224x224       | ????????         | 486,102          | 08/20/2020    |
| CEM4K-F3D     | 224x224x224   | ????????         | 3,618            | 11/21/2020    |
| CEM100K-S3D   | 224x224x32    | ????????         | 98,298           | 11/21/2020    |


## Download the Weights

```
wget https://somewebsite.com/{weight_id}
```

Weight IDs for versions of the unsupervised pretrained weights are:

| Model         | Dimension     | Weight ID        | Pretraining Data | Pretraining Algo. | Date Updated  | 
| ------------- | ------------- | ---------------- | ---------------- | ----------------- | ------------- |
| ResNet50      | 2D            | ????????         | CEM500K          | MoCoV2            | 08/20/2020    |


## Benchmarks

Mean IoU performance on each benchmark by weight ID:

| Weight ID     | All Mito  | CREMI S.C  | Guay   | Kasthuri++  | Lucchi++  | Perez  | UroCell  | Date Updated |
| ------------- | --------- | ---------- |------- | ----------- | --------- | -----  | -------- | ------------ |
| ???????       | 0.772     | 0.259      | 0.441  | 0.918       | 0.899     | 0.904  | 0.782    | 11/12/2020   |


## Citations


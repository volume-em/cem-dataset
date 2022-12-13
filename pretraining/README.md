# Pretraining

Before getting started download the latest CEM dataset from EMPIAR. At minimum you'll need access to a system
with 4 high-end GPUs (P100 or V100). Typically, pre-training takes 4-5 days (depending on the size of the dataset).

## SwAV

To run pretraining with SwAV, first update the ```data_path``` and ```model_path``` parameters in ```swav/swav_config.yaml```, then run:

```bash
python swav/train_swav.py swav_config.yaml
```

## MoCoV2

To run pretraining with MoCoV2, first update the ```data_path``` and ```model_path``` parameters in ```mocov2/mocov2_config.yaml```, then run:

```bash
python mocov2/train_mocov2.py mocov2_config.yaml
```



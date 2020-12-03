# Pretraining

## MoCoV2

To run pretraining you'll need to have downloaded the CellEMNet data. Update the data_file and model_dir parameters in the mocov2_config.yaml file. Then run:

```
python train_mocov2.py mocov2_config.yaml
```

The script was tested on machines with either 4 NVidia V100s or P100s. Runtime for a full 200 epochs on CEM500K is 3-4 days (faster if using V100s).
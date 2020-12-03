# Benchmark Evaluation

## Getting Started

Download and preprocess the benchmark datasets:

```
python setup_benchmarks/setup_data.py {save_dir}
```

Each benchmark has a corresponding .yaml config file in the benchmark_configs directory. The config files set directories and training and inference parameters. For now, you'll need to fill in the directories for each benchmark manually after downloading the datasets. (There are only 3 per config file: data_dir, model_dir, test_dir).

The finetune.py script handles all model training and result logging. The only required argument is the path to a config file:

```
python finetune.py benchmark_configs/all_mito.yaml
```

The finetune script outputs a state file which is saved in the model_dir defined in the config file. By default the file name is in the format:
{benchmark_name}-{pretraining}_ft_{finetune_layer}_epoch{epoch}_of_{total_epochs}.pth'
where epoch and total_epochs may refer to training iterations depending on the given learning rate policy (iterations for Poly and OneCycle, epochs for MultiStep).

To run inference, first determine the dimensionality of the test set (e.g. is the test set all 2D images or all 3D volumes?). Pick the inference script accordingly. Required arguments are a config file and a model state file. For example:

```
python inference2d.py perez_mito.yaml {model_dir}/perez_mito-cellemnet_mocov2_ft_none_epoch1000_of_1000.pth
python inference3d.py urocell.yaml {model_dir}/urocell-cellemnet_mocov2_ft_none_epoch1000_of_1000.pth
```

The one exception is the All Mitochondria benchmark in which the test set consists of both 2D and 3D data and a mishmash of evaluation protocols. To handle all these cases, the benchmark uses a custom inference script which can be run with:

```
python inference_mixed.py all_mito.yaml {model_dir}/all_mito-cellemnet_mocov2_ft_none_epoch1000_of_1000.pth
```

By default, all the config files are set with mlflow logging enabled. This will create a directory called mlruns. To launch the mlflow dashboard:

```
mlflow ui
```

The dashboard is grouped by benchmarks and records both training and inference results along with all the hyperparameters used for a run. This makes it easy to compare and reproduce results. For more details about mlflow see [here](https://mlflow.org/docs/latest/index.html).

## Snakemake Pipelines

The fastest way to evaluate results on the benchmarks is to use the provided [snakemake](https://snakemake.readthedocs.io/en/stable/) files. Simply run:

```
snakemake -s snakefile_wq
```

or 

```
snakemake -s snakefile_dq
```

We used snakefile_wq to train models with different initialization methods (random init., imagenet_supervised, cellemnet_mocov2) and snakefile_dq to compare models pretrained with the mocov2 algorithm on different EM datasets. 


## Adding new benchmarks

Adding a new benchmark is a cinch. Create a directory containing the benchmark data with the following structure:

```
{new_benchmark}
|__train
   |__images
   |__masks
|__valid (optional)
   |__images
   |__masks
|__test
   |__images
   |__masks
```

Make sure that pairs of images and masks have the same file names. All images and masks in the train and valid directories must be 2D images (.tiff, .png, .jpg, etc.). To convert from image/labelmap volumes to 2D images, use the setup_benchmarks/create_slices.py script (see script for parameters). Images in the test directory may be 2D or 3D ([any file type supported by SimpleITK is OK](https://simpleitk.readthedocs.io/en/master/IO.html)). Duplicate a config files from one of the other benchmarks and modify the directories and parameters as needed; make sure the name of the config file is {new_benchmark}.yaml.

To evaluate this new benchmark with snakemake, just add the benchmark name, {new_benchmark}, to either the BENCHMARKS2d or BENCHMARKS3d list based on the dimensionality of its test set. Then run the snakemake file as before.
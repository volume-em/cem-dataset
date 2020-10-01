import numpy as np
import os, sys, argparse, subprocess
import mlflow, torch
from glob import glob

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Evaluate on set of 2d images')
    parser.add_argument('weight_path', type=str, metavar='weight_path', help='Path to model state file')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    #get the arguments
    args = parse_args()
    weight_path = args['weight_path']
    
    #get the dataset and number of classes from the weight path
    dataset = weight_path.split('/')[-3]
    print(f'Using dataset {dataset}')
    
    #get the set of all models in the directory
    weight_paths = glob(os.path.join(weight_path, '*.pth'))
    print(f'Found {len(weight_paths)} weight files for eval')
    
    for wp in weight_paths:
        #decide what experiment we're working with
        if 'mocov2_imagenet' in wp:
            experiment = 'mocov2_imagenet'
        elif 'mocov2' in wp:
            experiment = 'mocov2'
        elif 'imagenet' in wp:
            experiment = 'imagenet'
        elif 'random_init' in wp:
            experiment = 'random_init'
        else:
            raise Exception(f'Weight path {wp} does not contain: mocov2, imagenet, or random_init!')

        print(f'Evaluating weight file {wp}, experiment {experiment}...')
        #load the weight file and get the mlflow run id
        run_id = torch.load(wp, map_location='cpu')['run_id']
        
        #start the mlflow run to log results
        with mlflow.start_run(run_id=run_id) as run:
            
            #if we already ran cross validation, skip it
            client = mlflow.tracking.MlflowClient()
            data = client.get_run(mlflow.active_run().info.run_id).data
            
            if dataset == 'guay':
                if 'guay_dense_gran_core_iou3d' in data.metrics:
                    print('Weight path already evaluated, skipping!')
                    continue
                    
                #lastly on guay
                impath = "/data/IASEM/conradrw/data/benchmarks/guay/volumes/guay_test.nrrd"
                mskpath = "/data/IASEM/conradrw/data/benchmarks/guay/labelmaps/guay_test.tif"
                save_path = "none"

                command = (
                    f"python /home/conradrw/nbs/moco_official/benchmark/evaluate2.5d.py {impath} {mskpath} {wp} {save_path} "
                    f"--num_classes 7 --exp {experiment} --model unet "
                    f"--axes 0 --mask_prediction"
                )

                with open('./temp.txt', mode = 'w+') as f:
                    subprocess.call(command.split(' '), stdout=f)

                f = open('./temp.txt', mode = 'r')
                string = '_'.join(f.readlines())
                iou_start = np.core.defchararray.find(string, 'Class IoUs')
                iou_end = np.core.defchararray.find(string, '\n', start=iou_start)
                class_ious = [float(iou) for iou in string[iou_start+11:iou_end].split(',')]
                class_names=["mito", "can_chan", "alpha_gran", "dense_gran", "dense_gran_core"]
                for iou,name in zip(class_ious, class_names):
                    mlflow.log_metric(f'guay_{name}_iou3d', iou, step=0)
                
            elif dataset == 'urocell':
                if 'urocell_mito_iou3d' in data.metrics:
                    print('Weight path already evaluated, skipping!')
                    continue
                    
                #then on urocell
                impath = "/data/IASEM/conradrw/data/benchmarks/urocell/volumes/fib1-0-0-0.nii.gz"
                mskpath = "/data/IASEM/conradrw/data/benchmarks/urocell/labelmaps/fib1-0-0-0.nii.gz"
                save_path = "none"

                command = (
                    f"python /home/conradrw/nbs/moco_official/benchmark/evaluate2.5d.py {impath} {mskpath} {wp} {save_path} "
                    f"--num_classes 3 --exp {experiment} --model unet "
                    f"--axes 0 1 2"
                )

                with open('./temp.txt', mode = 'w+') as f:
                    subprocess.call(command.split(' '), stdout=f)

                f = open('./temp.txt', mode = 'r')
                string = '_'.join(f.readlines())
                iou_start = np.core.defchararray.find(string, 'Class IoUs')
                iou_end = np.core.defchararray.find(string, '\n', start=iou_start)
                class_ious = [float(iou) for iou in string[iou_start+11:iou_end].split(',')]
                class_names = ["back", "lyso", "mito"]
                for iou,name in zip(class_ious, class_names):
                    mlflow.log_metric(f'urocell_{name}_iou3d', iou, step=0)
                    
            elif dataset == 'cremi':
                if 'cremi_iou3d' in data.metrics:
                    print('Weight path already evaluated, skipping!')
                    continue
                    
                #then on urocell
                impath = "/data/IASEM/conradrw/data/benchmarks/cremi/cremi_data/train/volumes/sample_C_20160501.nrrd"
                mskpath = "/data/IASEM/conradrw/data/benchmarks/cremi/cremi_data/train/labelmaps/sample_C_20160501_clefts.nrrd"
                save_path = "none"

                command = (
                    f"python /home/conradrw/nbs/moco_official/benchmark/evaluate2.5d.py {impath} {mskpath} {wp} {save_path} "
                    f"--num_classes 1 --exp {experiment} --model unet "
                    f"--axes 0"
                )

                with open('./temp.txt', mode = 'w+') as f:
                    subprocess.call(command.split(' '), stdout=f)

                f = open('./temp.txt', mode = 'r')
                string = '_'.join(f.readlines())
                iou_start = np.core.defchararray.find(string, 'IoU 3d')
                iou_end = np.core.defchararray.find(string, '\n', start=iou_start)
                iou = float(string[iou_start+7:iou_end])
                mlflow.log_metric('cremi_iou3d', iou, step=0)
            else:
                raise Exception(f'Dataset {dataset} not configured!')

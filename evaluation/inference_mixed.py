import os, argparse, yaml, subprocess
import numpy as np
import torch
import mlflow

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Evaluate performance on the all_mito benchmark')
    parser.add_argument('config', type=str, metavar='config', help='Path to a config yaml file')
    parser.add_argument('state_path', type=str, metavar='state_path', help='Path to model state file')
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['config'] = snakemake.input[0]
    params['state_path'] = snakemake.input[1]
    del params['_names']
    
    return params

if __name__ == '__main__':
    if 'snakemake' in globals():
        args = snakemake_args()
        #because snakemake expects an output file
        #we'll make a dummy file here
        with open(args['state_path'] + '.snakemake_all_mito', mode='w') as f:
            f.write("This is dummy file for snakemake")
    else:
        args = parse_args()
        
    script_dir = os.path.dirname(os.path.realpath(__file__))
        
    #read the config file
    config_path = args['config']
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    #add the state path to config
    config['state_path'] = args['state_path']
    state_path = config['state_path']
    
    #load the state_dict to extract out the mlflow run_id
    #(if there is one)
    state = torch.load(state_path, map_location='cpu')
    if 'run_id' in state:
        run_id = state['run_id']
    else:
        run_id = None
        
    del state
    
    #within the config file there should be lists
    #of parameters for both 2d and 3d datasets,
    #let's read them in
    test_dir = config['test_dir']
    benchmark_names = config['benchmark_names']
    benchmark_dimensions = config['benchmark_dimensions']
    eval_classes = config['eval_classes']
    thresholds = config['thresholds']
    instance_match2ds = config['instance_match2d']
    mode3ds = config['mode3d']
    mask_prediction3ds = config['mask_prediction3d']
    save_dirs = config['save_dirs']
    
    #loop through all of the benchmarks and run 
    #the appropriate inference script
    benchmark_ious = []
    for i in range(len(benchmark_names)):
        #select all the parameters from the benchmark
        benchmark = benchmark_names[i]
        benchmark_dim = benchmark_dimensions[i]
        eval_class = eval_classes[i]
        threshold = thresholds[i]
        instance_match2d = instance_match2ds[i]
        mode3d = mode3ds[i]
        mask_prediction3d = mask_prediction3ds[i]
        save_dir = save_dirs[i]
        
        #construct the test directory for the benchmark
        benchmark_dir = os.path.join(test_dir, f'{benchmark_dim}/{benchmark}/')
        
        #determine the script to run
        script = os.path.join(script_dir, f'inference{benchmark_dim}.py')
        
        #create a temporary config file to pass to the script
        config_dict = {}
        config_dict[f'test_dir{benchmark_dim}'] = benchmark_dir
        config_dict[f'threshold{benchmark_dim}'] = threshold
        config_dict[f'eval_classes{benchmark_dim}'] = [eval_class] #scripts expect a list
        config_dict[f'save_dir{benchmark_dim}'] = save_dir
        config_dict[f'mode{benchmark_dim}'] = mode3d
        config_dict[f'instance_match{benchmark_dim}'] = instance_match2d
        config_dict[f'mask_prediction{benchmark_dim}'] = mask_prediction3d
                
        tmp_yaml_file = f'{script_dir}/.tmp_mixed_config.yaml'
        with open(tmp_yaml_file, 'w') as f:
            yaml.dump(config_dict, f)
            
        #pass the config file and the state path 
        #to the inference script and store the stdout
        command = f'python {script} {tmp_yaml_file} {state_path}'
        result = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        print(result)
        
        if benchmark_dim == '2d':
            iou_start = np.core.defchararray.find(result, 'Mean IoU: ')
            iou_end = np.core.defchararray.find(result, '\n', start=iou_start)
            offset = len('Mean IoU: ')
            iou = float(result[iou_start+offset:iou_end])
        else:
            iou_start = np.core.defchararray.find(result, 'Overall mean IoU 3d: ')
            iou_end = np.core.defchararray.find(result, '\n', start=iou_start)
            offset = len('Overall mean IoU 3d: ')
            iou = float(result[iou_start+offset:iou_end])
            
        benchmark_ious.append(iou)
            
        print(f'{benchmark} IoU: {iou:.5f}')
        
        #if we're using logging, log the result by benchmark
        if run_id is not None:
            with mlflow.start_run(run_id=run_id) as run:
                mlflow.log_metric(f'{benchmark}_iou', iou, step=0)
                
    
    #and again, if logging calculate the mean over all the benchmarks
    #and save the results as Mean_IoU_All_Mito
    #if we're using logging, log the result by benchmark
    if run_id is not None:
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric(f'Test_Set_IoU', np.mean(benchmark_ious).item(), step=0)
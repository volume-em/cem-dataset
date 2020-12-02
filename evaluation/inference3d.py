import numpy as np
import os, argparse, cv2, yaml
import mlflow
import torch
import torch.nn as nn
import SimpleITK as sitk
from glob import glob

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

def factor_pad_tensor(tensor, factor=32):
    #takes a tensor and return the tensor
    #with reflection padding such that the
    #dimension of the padded tensor is divisible
    #by the given factor (32 for UNet)
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    return nn.ReflectionPad2d((0, pad_right, 0, pad_bottom))(tensor)

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Run orthoplane inference on volumes in a given directory')
    parser.add_argument('config', type=str, metavar='config', help='Path to a config yaml file')
    parser.add_argument('state_path', type=str, metavar='state_path', help='Path to model state file')
    parser.add_argument('--save_dir3d', type=str, metavar='save_dir3d',
                        help='Path to save segmentation result, if None, the results are not saved.')
    parser.add_argument('--mode3d', dest='mode3d', type=str, metavar='mode3d', choices=['orthoplane', 'stack'], default='stack',
                        help='Inference mode. Choice of orthoplane or stack.')
    parser.add_argument('--threshold3d', type=float, metavar='threshold3d', help='Prediction confidence threshold [0-1]')
    parser.add_argument('--eval_classes3d', dest='eval_classes3d', type=int, metavar='eval_classes', nargs='+',
                        help='Index/indices of classes to evaluate for multiclass segmentation')
    parser.add_argument('--mask_prediction3d', action='store_true', help='whether to evaluate IoU by first masking with ground truth')
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['config'] = snakemake.input[0]
    params['state_path'] = snakemake.input[1]
    del params['_names']
    
    #fill in the other arguments
    #that are handled by the config
    params['mode3d'] = None
    params['save_dir3d'] = None
    params['threshold3d'] = None
    params['eval_classes3d'] = None
    params['mask_prediction3d'] = False
    
    return params

if __name__ == '__main__':
    if 'snakemake' in globals():
        args = snakemake_args()
        #because snakemake expects an output file
        #we'll make a dummy file here
        with open(args['state_path'] + '.snakemake3d', mode='w') as f:
            f.write("This is dummy file for snakemake")
    else:
        args = parse_args()
        
    #read the config file
    with open(args['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    #add the state path to config
    config['state_path'] = args['state_path']
    
    #overwrite the parameters, if given
    if args['save_dir3d'] is not None:
        config['save_dir3d'] = args['save_dir3d']
    if args['mode3d'] is not None:
        config['mode3d'] = args['mode3d']
    if args['threshold3d'] is not None:
        config['threshold3d'] = args['threshold3d']
    if args['eval_classes3d'] is not None:
        config['eval_classes3d'] = args['eval_classes3d']
    if args['mask_prediction3d'] is not False:
        config['mask_prediction3d'] = args['mask_prediction3d']

    #read in the arguments
    test_dir = config['test_dir3d']
    state_path = config['state_path']
    save_dir = config['save_dir3d']
    mode = config['mode3d']
    threshold = config['threshold3d']
    
    #make sure that the threshold is set
    #to default if something else wasn't given
    if threshold is None:
        threshold = 0.5
    
    eval_classes = config['eval_classes3d']
    mask_prediction = config['mask_prediction3d']
    
    #if we're going to save the segmentation, let's
    #make sure that the directory exists
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Created directory {save_dir}')
            
    #we expect the test directory to contain subdirectories
    #called images and masks
    impaths = glob(os.path.join(test_dir, 'images/*'))
    
    #if there's no masks directory then we only run inference
    #and will not compare predictions against a ground truth
    inference_only = True
    if os.path.isdir(os.path.join(test_dir, 'masks')):
        inference_only = False
    
    #load the model state file
    state = torch.load(state_path, map_location='cpu')
    
    #assuming we used logging during training, get the
    #mlflow run_id from state. this allows us to log the
    #calculated IoU results with the model run
    if 'run_id' in state:
        run_id = state['run_id']
    else:
        run_id = None
    
    #there are a few parameters that we need to extract from
    #the model state
    #1. What are the norms? Saved in the state file
    #2. How many input channels? Get it from the length of the norms
    #3. How many output channels? Get it from size of the last
    #parameter in the state_dict (the output bias tensor)
    #norms = state['norms']
    norms = [0.58331613, 0.09966064]
    
    #if there are multiple channels, the mean and std will
    #be lists, otherwise their just single floats
    #gray channels is the same as input channels
    gray_channels = len(norms[0]) if hasattr(norms[0], '__len__') else 1
    num_classes = list(state['state_dict'].values())[-1].size(0) #same as output channels
    
    #create the evaluation transforms with
    #the correct normalization
    eval_tfs = Compose([
        Normalize(mean=norms[0], std=norms[1]),
        ToTensorV2()
    ])
    
    #create the UNet, at least for now, we're only supporting ResNet50
    model = smp.Unet('resnet50', in_channels=gray_channels, encoder_weights=None, classes=num_classes)
    model.load_state_dict(state['state_dict'])
    
    #again, we're assuming that there's access to a GPU
    model = model.cuda()
    
    #set the model to eval mode
    model.eval()
    
    #define the planes that we'll run inference over
    #if mode is stack we only run over axis 0 (yx plane)
    #if mode is orthoplane we run over axes 0, 1, 2 (yx, xz, zy)
    #NOTE: numpy arrays are (depth, height, width)
    #SimpleITK images are (width, height, depth)
    if mode == 'stack':
        axes = [0]
    elif mode == 'orthoplane':
        axes = [0, 1, 2]
    else:
        raise Exception(f'Inference mode must be orthoplane or stack, got {mode}')
    
    #set scaling factor based on number of axes
    #instead of averaging the results from predictions
    #over multiple planes, we just add together the
    #scaled versions. This let's us operate with unsigned
    #8-bit integer voxels: a great memory savings over float32
    scaling = 255 / len(axes)
    
    #if we're working with a single class
    #use the threshold, otherwise take the argmax
    threshold = int(255 * threshold)
    
    #loop through all of the image paths and make predictions
    volume_mean_ious = []
    for imvol in impaths:
        print(f'Loading {imvol}')
        #load the image volume for inference
        orig_vol = sitk.ReadImage(imvol)

        #make a separate variable for numpy copy
        #we'll need to resample the prediction volume
        #to match the size and space of the original
        #volume (i.e. restore the correct metadata)
        im_vol = sitk.GetArrayFromImage(orig_vol)
        print(f'Volume size {im_vol.shape}')

        #assert that the volume type is uint8
        assert(im_vol.dtype == np.uint8), \
        'Image volume must have 8-bit unsigned voxels!'
        
        #alright, now we're ready to run inference
        #let's create an empty prediction volume to
        #store the segmentations
        prediction_volume = np.zeros((num_classes, *im_vol.shape), dtype=np.uint8) #e.g. (3, 256, 256, 256)

        #loop over the axes, predict, and store segmentations
        for ax in axes:
            print(f'Predicting over axis {ax}')

            #create a stack of images sliced from the current axis
            #over which to predict
            stack = np.split(im_vol, im_vol.shape[ax], axis=ax)
            for index, image in enumerate(stack):
                #make sure the image has the correct number of 
                #grayscale channels: ImageNet models have 3
                if gray_channels == 3:
                    image = cv2.cvtColor(np.squeeze(image), cv2.COLOR_GRAY2RGB)
                else:
                    #add an empty channel dim
                    image = np.squeeze(image)[..., None]

                #at this point the image size is either (H, W, 3) or (H, W, 1)
                #after augmentation it is (1, GRAY_CHANNELS, H, W)
                image = eval_tfs(image=image)['image'].unsqueeze(0)

                #load image to gpu
                image = image.cuda()

                #get the image size and pad the image to a factor
                h, w = image.size()[2:]
                image = factor_pad_tensor(image, factor=32)

                with torch.no_grad():
                    #make the prediction and remove the padding
                    prediction = model(image)[..., :h, :w]

                    #the number of segmentation classes determines if we use
                    #a Sigmoid or a SoftMax on the output
                    if num_classes == 1:
                        prediction = nn.Sigmoid()(prediction) #(1, 1, H, W)
                    else:
                        prediction = nn.Softmax(dim=1)(prediction) #(1, NC, H, W)

                    #now we can convert the prediction to a numpy array of uint8 type
                    #if in orthoplane inference the max value of the segmentation will be 85
                    #otherwise, for stack inference, it will be 255
                    prediction = (prediction.squeeze(0).detach().cpu().numpy() * scaling).astype(np.uint8) #(NC, H, W)

                    #the last step is to add the slice prediction into the prediction
                    #volume. for orthoplane, 85 + 85 + 85 = 255.
                    if ax == 0:
                        prediction_volume[:, index] += prediction
                    elif ax == 1:
                        prediction_volume[:, :, index] += prediction
                    else:
                        prediction_volume[:, :, :, index] += prediction
            
        if num_classes == 1:
            prediction_volume = (prediction_volume > threshold).astype(np.uint8)[0] #(1, D, H, W) --> (D, H, W)
        else:
            prediction_volume = np.argmax(prediction_volume, axis=0).astype(np.uint8) #(NUM_CLASSES, D, H, W) --> (D, H, W)

        #if we're saving the prediction, now's the time to do it
        if save_dir is not None:
            #convert the numpy volume to a SimpleITK image
            save_prediction = sitk.GetImageFromArray(prediction_volume)

            #copy the metadata from the original volume
            #this include pixel spacing, origin, and direction
            save_prediction.CopyInformation(orig_vol)
            
            #get the volume name from it's path
            vol_name = imvol.split('/')[-1]

            #save the volume
            sitk.WriteImage(save_prediction, os.path.join(save_dir, vol_name))
            del save_prediction

        #this next section only applies if there is a ground truth volume
        #to compare against
        if not inference_only:
            #load the equivalent volume from masks
            vol_name = imvol.split('/')[-1]
            gtvol = os.path.join(test_dir, f'masks/{vol_name}')
            
            if not os.path.isfile(gtvol):
                print(f'No ground truth found at {gtvol} for {imvol}')
            else:
                #load the mask volume as a numpy array of 8-bit unsigned int
                gtvolume = sitk.GetArrayFromImage(sitk.ReadImage(gtvol))
                gtvolume = gtvolume.astype(np.uint8)

                #in some cases, the entire gtvolume may not be labeled
                #this is the case for the Guay benchmark. in that case
                #we need to ignore any predictions that fall outside
                #the labeled region in the gtvolume
                if mask_prediction:
                    mask = gtvolume > 0
                    prediction_volume *= mask

                #if we defined eval classes, then we'll only
                #consider them during evaluation. otherwise we'll
                #consider all labels
                if eval_classes is None:
                    if num_classes == 1:
                        eval_classes = [1]
                    else:
                        eval_classes = list(range(1, num_classes))

                #loop over each of the eval_classes and
                #calculate the IoUs
                class_ious = []
                for label in eval_classes:
                    label_pred = prediction_volume == label
                    label_gt = gtvolume == label
                    intersect = np.logical_and(prediction_volume == label, gtvolume == label).sum()
                    union = np.logical_or(prediction_volume == label, gtvolume == label).sum()

                    #add small epsilon to prevent zero division
                    iou = (intersect + 1e-5) / (union + 1e-5)

                    #print the class IoU
                    print(f'Class {label} IoU 3d: {iou}')

                    #store the result
                    class_ious.append(intersect / union)

                #calculate and print the mean iou
                mean_iou = np.mean(class_ious).item()
                print(f'Mean IoU 3d: {mean_iou}')
                
            volume_mean_ious.append(mean_iou)
            
    if not inference_only:
        mean_iou = np.mean(volume_mean_ious).item()
        print(f'Overall mean IoU 3d: {mean_iou}')

        #all that's left now is to store the mean iou
        #if run_id is not None:
        #    with mlflow.start_run(run_id=run_id) as run:
        #        mlflow.log_metric('Test_Set_IoU', mean_iou, step=0)

        #    print('Stored mean IoU in mlflow run.')
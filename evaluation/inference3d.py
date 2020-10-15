import numpy as np
import os, argparse, cv2
import mlflow
import torch
import torch.nn as nn
import SimpleITK as sitk

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
    parser = argparse.ArgumentParser(description='Evaluate on a 3d volume')
    parser.add_argument('imvol', type=str, metavar='imvol', help='Path to image volume')
    parser.add_argument('state_path', type=str, metavar='state_path', help='Path to model state file')
    parser.add_argument('--out', type=str, metavar='save_path', dest='save_path',
                        help='Path to save segmentation result, if None, the results are not saved.')
    parser.add_argument('--mode', dest='mode', type=str, metavar='mode', choices=['orthoplane', 'stack'], default='stack',
                        help='Inference mode. Choice of orthoplane or stack.')
    parser.add_argument('--threshold', type=float, metavar='threshold', default=0.5,
                        help='Prediction confidence of threshold')
    parser.add_argument('--gtvol', type=str, metavar='gtvol', help='Path to a ground truth labelmap volume')
    parser.add_argument('--eval_classes', dest='eval_classes', type=int, metavar='eval_classes', nargs='+',
                        help='Index/indices of classes to evaluate for multiclass segmentation')
    parser.add_argument('--mask_prediction', action='store_true', help='mask prediction with ground truth')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    #read in the arguments
    args = parse_args()
    imvol = args['imvol']
    state_path = args['state_path']
    save_path = args['save_path']
    mode = args['mode']
    threshold = args['threshold']
    
    #the last arguments are only if there is a ground truth
    #against which to compare the prediction
    gtvol = args['gtvol']
    eval_classes = args['eval_classes']
    mask_prediction = args['mask_prediction']
    
    #if we're going to save the segmentation, let's
    #make sure that the path exists
    if save_path is not None:
        n_path_dirs = len(save_path.split('/'))
        if n_path_dirs > 1:
            save_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
                print(f'Created directory {save_dir}')
    
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
    #1. What are the norms? Saved in the state dict
    #2. How many input channels? Get it from the length of the norms
    #3. How many output channels? Get it from size of the last
    #parameter in the state_dict (the output bias tensor)
    norms = state['norms']
    
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
    
    #alright, now we're ready to run inference
    #let's create an empty prediction volume to
    #store the segmentations
    prediction_volume = np.zeros((num_classes, *im_vol.shape), dtype=np.uint8) #e.g. (3, 256, 256, 256)
    
    #define the planes that we'll run inference over
    #if mode is stack we only run over axis 0 (yx plane)
    #if mode is orthoplane we run over axes 0, 1, 2 (yx, xz, zy)
    #NOTE: numpy arrays are (depth, height, width)
    #SimpleITK images are (width, height, depth)
    axes = [0]
    if mode == 'orthoplane':
        axes.extend([1, 2])
    
    #set scaling factor based on number of axes
    #instead of averaging the results from predictions
    #over multiple planes, we just add together the
    #scaled versions. This let's us operate with unsigned
    #8-bit integer voxels: a great memory savings over float32
    scaling = 255 / len(axes)
    
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
                #if in orthoplane inference the max value of the segmentation will b 85
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
                    
    #if we're working with a single class
    #use the threshold, otherwise take the argmax
    threshold = int(255 * threshold)
    if num_classes == 1:
        prediction_volume = (prediction_volume > threshold).astype(np.uint8)[0] #(1, D, H, W) --> (D, H, W)
    else:
        prediction_volume = np.argmax(prediction_volume, axis=0).astype(np.uint8) #(NUM_CLASSES, D, H, W) --> (D, H, W)
        
    #if we're saving the prediction, now's the time to do it
    if save_path is not None:
        #convert the numpy volume to a SimpleITK image
        prediction_volume = sitk.GetImageFromArray(prediction_volume)
        
        #copy the metadata from the original volume
        #this include pixel spacing, origin, and direction
        prediction_volume.CopyInformation(orig_vol)
        
        #save the volume
        sitk.WriteImage(prediction_volume, save_path)
    
    #this next section only applies if there is a ground truth volume
    #to compare against
    if gtvol is not None:
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
        #consider all labels that appear in gtvol
        if eval_classes is None:
            #get the unique labels in gtvolume
            #always excluding 0 (background)
            eval_classes = np.unique(gtvolume)[1:]
            
        #loop over each of the eval_classes and
        #calculate the IoUs
        class_ious = []
        for label in eval_classes:
            label_pred = prediction_volume == label
            label_gt = gtvolume == label
            intersect = np.logical_and(prediction_volume == label, gtvolume == label).sum()
            union = np.logical_or(prediction_volume == label, gtvolume == label).sum()
            
            #because we're evaluating exclusively over labels that exist
            #in the volume, we know that union is always >0
            iou = intersect / union
            
            #print the class IoU
            print(f'Class {label} IoU 3d: {iou}')
            
            #store the result
            class_ious.append(intersect / union)
            
        #calculate and print the mean iou
        mean_iou = np.mean(class_ious).item()
        print(f'Mean IoU 3d: {mean_iou}')
            
        #all that's left now is to store the mean iou
        if run_id is not None:
            with mlflow.start_run(run_id=run_id) as run:
                mlflow.log_metric('Mean_IoU_3d', mean_iou, step=0)
            print('Stored mean IoU in mlflow run.')
import os, sys, argparse, warnings, cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage import measure
from skimage.transform import resize
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from data import SegmentationData, FactorResize

def mean_iou(output, target):
    #make target the same shape as output by unsqueezing
    #the channel dimension, if needed
    if target.ndim == output.ndim - 1:
        target = target.unsqueeze(1)

    #get the number of classes from the output channels
    n_classes = output.size(1)

    #get reshape size based on number of dimensions
    #can exclude first 2 dims, which are always batch and channel
    empty_dims = (1,) * (target.ndim - 2)

    if n_classes > 1:
        #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
        k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
        target = (target == k)

        #softmax the output
        output = nn.Softmax(dim=1)(output)
    else:
        #just sigmoid the output
        output = (nn.Sigmoid()(output) > 0.5).long()

    #cast target to the correct type for operations
    target = target.type(output.dtype)

    #multiply the tensors, everything that is still as 1 is part of the intersection
    #(N,)
    dims = (0,) + tuple(range(2, target.ndim))
    intersect = torch.sum(output * target, dims)

    #compute the union, (N,)
    union = torch.sum(output + target, dims) - intersect

    #avoid division errors by adding a small epsilon
    iou = (intersect + 1e-7) / (union + 1e-7)

    return iou.mean().item()

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Evaluate on set of 2d images')
    parser.add_argument('data_dir', type=str, metavar='data_dir', help='Directory containing 2d test images')
    parser.add_argument('state_path', type=str, metavar='state_path', help='Path to model state file')
    parser.add_argument('--save_dir', type=str, metavar='save_dir', dest='save_dir',
                        help='Directory to save segmentation results, if None, the results are not saved.')
    parser.add_argument('--threshold', type=float, metavar='threshold', default=0.5,
                        help='Prediction confidence threshold [0-1]')
    parser.add_argument('--eval_classes', dest='eval_classes', type=int, metavar='eval_classes', nargs='+',
                        help='Index/indices of classes to evaluate for multiclass segmentation')
    parser.add_argument('--instance_match', action='store_true', help='whether to evaluate IoU by instance matching')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    #read in the arguments
    args = parse_args()
    data_dir = args['data_dir']
    state_path = args['state_path']
    save_dir = args['save_dir']
    threshold = args['threshold']
    
    #the last arguments are only if there is a ground truth
    #against which to compare the prediction
    eval_classes = args['eval_classes']
    instance_match = args['instance_match']
    
    #if we're going to save the segmentation, let's
    #make sure that the directory exists
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Created directory {save_dir}')
            
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
    #norms = state['norms']
    norms = [0.58331613, 0.09966064]
    
    #if there are multiple channels, the mean and std will
    #be lists, otherwise their just single floats
    #gray channels is the same as input channels
    gray_channels = len(norms[0]) if hasattr(norms[0], '__len__') else 1
    num_classes = list(state['state_dict'].values())[-1].size(0) #same as output channels
    
    #create the evaluation transforms with the correct normalization
    #and use FactorResize to resize images such that height and width 
    #are divisible by 32
    eval_tfs = Compose([
        FactorResize(32),
        Normalize(mean=norms[0], std=norms[1]),
        ToTensorV2()
    ])

    #create the dataset and dataloader
    test_data = SegmentationData(data_dir, tfs=eval_tfs, gray_channels=gray_channels)
    test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    
    #determine if we're in inference only mode or not
    inference_only = False if test_data.has_masks else True
    
    #create the UNet. at least for now we're only supporting ResNet50
    model = smp.Unet('resnet50', in_channels=gray_channels, encoder_weights=None, classes=num_classes)
    model.load_state_dict(state['state_dict'])
    
    #again, we're assuming that there's access to a GPU
    model = model.cuda()
    
    #set the model to eval mode
    model.eval()
    
    #loop over the images, predict, save, compute iou
    image_ious = []
    for data in test:
        #load image to gpu
        image = data['image'].cuda(non_blocking=True) #(1, 1, H', W')
            
        #filename and original image shape before resizing
        fname = data['fname'][0]
        shape = data['shape']
        
        #run inference
        with torch.no_grad():
            #make the prediction
            prediction = model(image).detach()
            
            #apply sigmoid or softmax based on num_classes
            if num_classes == 1:
                #(1, 1, H, W) --> (H, W)
                prediction = (nn.Sigmoid()(prediction) > threshold).squeeze().cpu().numpy()
            else:
                #for multiclass we apply softmax and take the class with
                #the highest probability
                prediction = nn.Softmax(dim=1)(prediction) #(1, C, H, W)
                prediction = torch.argmax(prediction, dim=1) #(1, H, W)
                prediction = prediction.squeeze().cpu().numpy() #(H, W)
        
        #resize the prediction to original shape
        #order=0 means nearest neighbor interpolation
        prediction = resize(prediction, shape, preserve_range=True, order=0).astype(np.uint8)
        
        #save the prediction with the correct filename
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, fname), prediction)
        
        if not inference_only:
            #the mask can stay on cpu, if there is one
            #notice that the H,W will be different because
            #the mask is not resized by FactorResize
            mask = data['mask'].squeeze().numpy()  #(1, H, W) --> (H, W)
        
            #now for the last step: calculate the iou
            #determine all of the eval_classes
            #always ignoring 0 (background)
            if eval_classes is None:
                labels = np.unique(mask)[:1]
            else:
                labels = eval_classes

            class_ious = []
            for label in labels:
                #only consider areas in the mask and prediction
                #corresponding to the current label
                label_mask = mask == label
                label_pred = prediction == label

                #there's 1 hiccup to consider that occurs
                #when not all of the instances of an object
                #are labeled in a 2d image. this is the case for all
                #the datasets in the Perez benchmark (e.g. there may be
                #20 mitochondria in an image, but only 10 were labeled).
                #to handle this case we'll need to only consider parts of the
                #prediction that have some overlap with the ground truth. in
                #principle this will hide some instances that are FP, but it will
                #not hide FP pixels that were predicted as part of an instance
                if instance_match:
                    #because we're working with labeled instances
                    #we will label each connected component in the mask
                    #as a separate object
                    instance_label_mask = measure.label(label_mask)
                    instance_label_pred = measure.label(label_pred)

                    #we're going to evaluate IoU over all the pixels for
                    #the current label within the image
                    instance_matched_prediction = np.zeros_like(label_pred)
                    mask_instance_labels = np.unique(instance_label_mask)[1:]
                    for mask_instance_label in mask_instance_labels:
                        #find all the instance labels in the prediction that 
                        #coincide with the current mask instance label
                        prediction_instance_labels = np.unique(
                            instance_label_pred[instance_label_mask == mask_instance_label]
                        )

                        #add all pixels in the prediction with the detected instance
                        #labels to the instance_matched_prediction
                        for prediction_instance_label in prediction_instance_labels:
                            if prediction_instance_label != 0: #ignore background
                                instance_matched_prediction += instance_label_pred == prediction_instance_label

                    #alright, now finally, we can compare the label_mask and the instance_matched_prediction
                    intersect = np.logical_and(instance_matched_prediction, label_mask).sum()
                    union = np.logical_or(instance_matched_prediction, label_mask).sum()
                else:
                    #this is the case that we hope to find ourselves in.
                    #evaluation is much easier here
                    intersect = np.logical_and(label_pred, label_mask).sum()
                    union = np.logical_or(label_pred, label_mask).sum()

                class_ious.append((intersect + 1e-5) / (union + 1e-5))

            image_ious.append(class_ious)

    #report the mean IoU
    if not inference_only:
        image_ious = np.array(image_ious)
        mean_class_ious = image_ious.mean(axis=0)
        for ix, mci in enumerate(mean_class_ious):
            print(f'Class {ix} IoU 2d: {mci}')
            
        #print the overall mean
        mean_iou = image_ious.mean()
        print(f'Mean IoU {mean_iou:.5f}')
        
        #store the results if logging in mlflow
        if run_id is not None:
            with mlflow.start_run(run_id=run_id) as run:
                mlflow.log_metric('Mean_IoU_2d', mean_iou, step=0)
            print('Stored mean IoU in mlflow run.')
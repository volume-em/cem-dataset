import numpy as np
import os, sys, argparse, warnings
import torch
import torch.nn as nn
import SimpleITK as sitk
import skimage.measure as measure
from skimage import io
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/home/conradrw/nbs/moco_official/')
from moco.resnet import resnet50 as moco_resnet50
from torchvision.models import resnet50

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

sys.path.append('/home/conradrw/nbs/mitonet/')
from model2d.deeplab import DeepLabV3
from model2d.data import MitoData, FactorResize

import segmentation_models_pytorch as smp

MOCO_NORMS = {
    'filtered': [0.58331613, 0.09966064],
    'deduped': [0.640966126075197, 0.09368614112269238],
    'unfiltered': [0.69583, 0.09672],
    'bloss': [0.62337948, 0.13348031]
}

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
    parser.add_argument('data_dir', type=str, metavar='data_path', help='Path to image volume')
    parser.add_argument('weight_path', type=str, metavar='weight_path', help='Path to model state file')
    #parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save segmentation result')
    parser.add_argument('--model', type=str, metavar='model', help='Name of model, unet or deeplab')
    parser.add_argument('--num_classes', type=int, metavar='num_classes', help='Path to image volume')
    parser.add_argument('--exp', type=str, metavar='exp', help='Experiment type',
                        choices=['mocov2', 'imagenet', 'random_init', 'mocov2_imagenet'])
    parser.add_argument('--mask_prediction', action='store_true', help='mask prediction with ground truth')
    parser.add_argument('--model_class', type=int, metavar='model_class', default=-1,
                        help='Class index from num_classes to evaluate, -1 means evaluate all')
    parser.add_argument('--mask_class', type=int, metavar='mask_class', default=-1,
                        help='Class index for mask to evaluate, -1 means evaluate all')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    #get the arguments
    args = parse_args()
    data_dir = args['data_dir']
    weight_path = args['weight_path']
    model_name = weight_path.split('/')[-1].split('.pth')[0]
    #save_path = args['save_path']
    num_classes = args['num_classes']
    experiment = args['exp']
    model_class = args['model_class']
    mask_class = args['mask_class']
    
    if model_class > -1:
        assert(mask_class > -1), 'Mask class cannot be -1 if model class is not -1'
    elif mask_class > -1:
        assert(model_class > -1), 'Model class cannot be -1 if mask class is not -1'
    
    #if not os.path.isdir(save_path):
    #    os.mkdir(save_path)
    
    if experiment == 'mocov2' or experiment == 'random_init':
        gray_channels = 1
        weight_prep = weight_path.split('_')[-1][:-4]
        moco_mean, moco_std = MOCO_NORMS[weight_prep]
        normalize = Normalize(mean=[moco_mean], std=[moco_std])
        resnet = moco_resnet50()
    elif experiment == 'imagenet' or experiment == 'mocov2_imagenet':
        gray_channels = 3
        normalize = Normalize()
        resnet = resnet50()
    else:
        raise Exception('Experiment {} not supported!'.format(experiment))

    eval_tfs = Compose([
        normalize,
        ToTensorV2()
    ])

    test_data = MitoData(data_dir, tfs=eval_tfs, gray_channels=gray_channels)
    test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    
    if args['model'] == 'deeplab':
        model = DeepLabV3(resnet, num_classes)
    elif args['model'] == 'unet':
        model = smp.Unet('resnet50', in_channels=gray_channels, encoder_weights=None, classes=args['num_classes'])
        
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['state_dict'])
    model = model.to('cuda:0')
    
    #loop over the images, pad, predict and save predictions
    ious = []
    for data in test:
        #load image to gpu
        image = data['im'].to('cuda:0') #(B, 1, H, W)
        mask = data['msk'].to('cuda:0') #(B, H, W)
        
        #get the image size and calculate the required padding
        h, w = image.size()[2:]
        pad_bottom = 32 - h % 32 if h % 32 != 0 else 0
        pad_right = 32 - w % 32 if w % 32 != 0 else 0
        image = nn.ReflectionPad2d((0, pad_right, 0, pad_bottom))(image)
        
        #evaluate
        with torch.no_grad():
            prediction = model.eval()(image)
            prediction = prediction[..., :h, :w] #remove the padding
            
            mask = mask.squeeze().cpu().numpy()
            if num_classes == 1:
                labeled_pred = (nn.Sigmoid()(prediction) > 0.5).squeeze().cpu().numpy()
            else:
                labeled_pred = nn.Softmax(dim=1)(prediction) #(1, C, H, W)
                labeled_pred = torch.argmax(prediction, dim=1) #(1, H, W)
                labeled_pred = labeled_pred.squeeze().cpu().numpy()
            
            #calculate the mean iou
            if args['mask_prediction']:
                if mask_class > -1:
                    labeled_mask = measure.label(mask == mask_class)
                else:
                    labeled_mask = measure.label(mask > 0)
                
                if num_classes > 1 and model_class > -1:
                    labeled_pred = labeled_pred == model_class
                elif num_classes > 1:
                    raise Exception(f'Model class cannot be -1 if mask_prediction and num_classes > 1 are True!')

                labeled_pred = measure.label(labeled_pred)
                
                total_intersect = 0
                total_union = 0
                for l in np.unique(labeled_mask)[1:]:
                    lmask = labeled_mask == l
                    plabels = np.unique(labeled_pred[lmask])[1:]
                    lpred = np.zeros_like(labeled_pred)
                    for pl in plabels:
                        lpred += labeled_pred == pl

                    total_intersect += np.sum(lmask * lpred)
                    total_union += np.sum(lmask) + np.sum(lpred)

                iou = (total_intersect + 1e-7) / (total_union - total_intersect + 1e-7)
                ious.append(iou)
            else:
                labels = list(range(1, num_classes + 1))

                if model_class > -1 and mask_class > -1:
                    pred_labels = [model_class]
                    mask_labels = [mask_class]
                else:
                    pred_labels = labels
                    mask_labels = labels

                #print(f'Evaluating mask class(es) {mask_labels}...')
                for pl, ml in zip(pred_labels, mask_labels):
                    label_pred = labeled_pred == pl
                    label_gt = mask == ml
                    intersect = (label_pred * label_gt).sum()
                    union = label_pred.sum() + label_gt.sum() - intersect
                    iou = (intersect + 1e-7) / (union + 1e-7)

                    ious.append(iou)

                #ious.append(mean_iou(prediction, mask))
            
            #if num_classes == 1:
            #    prediction = nn.Sigmoid()(prediction) > 0.5 #(B, 1, H, W)
            #else:
            #    prediction = nn.Softmax(dim=1)(prediction)
            #    prediction = torch.argmax(prediction, dim=1) #(B, H, W)
            
            #prediction = prediction.squeeze().detach().cpu().numpy().astype(np.uint8) #(H, W)
        
        #fname = data['fname'][0] #(1, 1) --> (1,)
        
        #save the prediction
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    io.imsave(os.path.join(save_path, f'{fname}'), prediction)
        
    #report the mean IoU
    ious = np.array(ious)
    print(f'Mean IoU {ious.mean():.5f}')
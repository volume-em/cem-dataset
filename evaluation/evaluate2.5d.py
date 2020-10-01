import numpy as np
import os, sys, argparse, warnings, cv2
import torch
import torch.nn as nn
import SimpleITK as sitk
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
from model2d.data import MitoData

import segmentation_models_pytorch as smp

MOCO_NORMS = {
    'filtered': [0.58331613, 0.09966064],
    'deduped': [0.640966126075197, 0.09368614112269238],
    'unfiltered': [0.69583, 0.09672],
    'bloss': [0.62337948, 0.13348031],
    'dice': [0.58331613, 0.09966064],
}

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Evaluate on set of 2d images')
    parser.add_argument('imvol', type=str, metavar='imvol', help='Path to image volume')
    parser.add_argument('mskvol', type=str, metavar='mskvol', help='Path to mask volume')
    parser.add_argument('weight_path', type=str, metavar='weight_path', help='Path to model state file')
    parser.add_argument('save_path', type=str, metavar='save_path', 
                        help='Path to save segmentation result')
    parser.add_argument('--model', type=str, metavar='model', help='Name of model, unet or deeplab')
    parser.add_argument('--num_classes', type=int, metavar='num_classes', help='Path to image volume')
    parser.add_argument('--exp', type=str, metavar='exp', help='Experiment type',
                        choices=['mocov2', 'imagenet', 'random_init', 'mocov2_imagenet'])
    parser.add_argument('--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-yz, 1-xz, 2-xy)')
    parser.add_argument('--mask_prediction', action='store_true', help='mask prediction with ground truth')
    parser.add_argument('--model_class', type=int, metavar='model_class', default=-1,
                        help='Class index from num_classes to evaluate, -1 means evaluate all')
    parser.add_argument('--mask_class', type=int, metavar='mask_class', default=-1,
                        help='Class index for mask to evaluate, -1 means evaluate all')
    parser.add_argument('--threshold', type=float, metavar='threshold', default=0.5,
                        help='Prediction confidence of threshold')
    args = vars(parser.parse_args())
    
    return args

if __name__ == '__main__':
    #get the arguments
    args = parse_args()
    imvol = args['imvol']
    mskvol = args['mskvol']
    weight_path = args['weight_path']
    model_name = weight_path.split('/')[-1].split('.pth')[0]
    save_path = args['save_path']
    num_classes = args['num_classes']
    experiment = args['exp']
    axes = args['axes']
    mask_prediction = args['mask_prediction']
    model_class = args['model_class']
    mask_class = args['mask_class']
    threshold = args['threshold']
    
    if model_class > -1:
        assert(mask_class > -1), 'Mask class cannot be -1 if model class is not -1'
    elif mask_class > -1:
        assert(model_class > -1), 'Model class cannot be -1 if mask class is not -1'
    
    #load the image and mask volumes
    orig_vol = sitk.ReadImage(imvol)
    im_vol = sitk.GetArrayFromImage(orig_vol)
    print(f'Volume size {im_vol.shape}')
    
    #if not os.path.isdir('/'.join(save_path.split('/')[:-1])):
    #    os.mkdir('/'.join(save_path.split('/')[:-1]))
    
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
    
    if args['model'] == 'deeplab':
        model = DeepLabV3(resnet, num_classes)
    elif args['model'] == 'unet':
        model = smp.Unet('resnet50', in_channels=gray_channels, encoder_weights=None, classes=args['num_classes'])
        
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['state_dict'])
    model = model.to('cuda:0')
    
    #loop over the images, pad, predict and save predictions
    ious = []
    prediction_volume = np.zeros((num_classes, *im_vol.shape), dtype=np.uint8)
    scaling = 255 / len(axes)
    
    for ax in axes:
        print('Predicting over axis ', ax)
        stack = np.split(im_vol, im_vol.shape[ax], axis=ax)
        for index, image in enumerate(stack):
            if gray_channels == 3:
                image = cv2.cvtColor(np.squeeze(image), cv2.COLOR_GRAY2RGB)
            else:
                #add an empty channel dim
                image = np.squeeze(image)[..., None]
                
            #apply the augmentations and add the batch dimension
            image = eval_tfs(image=image)['image'].unsqueeze(0)
            
            #load image to gpu
            image = image.to('cuda:0') #(1, 1, H, W)

            #get the image size and calculate the required padding
            h, w = image.size()[2:]
            pad_bottom = 32 - h % 32 if h % 32 != 0 else 0
            pad_right = 32 - w % 32 if w % 32 != 0 else 0
            image = nn.ReflectionPad2d((0, pad_right, 0, pad_bottom))(image)

            #evaluate
            with torch.no_grad():
                prediction = model.eval()(image)
                prediction = prediction[..., :h, :w] #remove the padding

                if num_classes == 1:
                    prediction = nn.Sigmoid()(prediction) #(1, 1, H, W)
                else:
                    prediction = nn.Softmax(dim=1)(prediction) #(1, NC, H, W)

                prediction = (prediction.squeeze(0).detach().cpu().numpy() * scaling).astype(np.uint8) #(NC, H, W)
                if ax == 0:
                    prediction_volume[:, index] += prediction
                elif ax == 1:
                    prediction_volume[:, :, index] += prediction
                else:
                    prediction_volume[:, :, :, index] += prediction
                    
    #if we're working with a single class
    #use the threshold
    threshold = int(255 * threshold)
    if num_classes == 1:
        prediction_volume = (prediction_volume > threshold).astype(np.uint8)[0]
    else:
        prediction_volume = np.argmax(prediction_volume, axis=0).astype(np.uint8)
        
    print(f'Prediction volume {prediction_volume.shape} of {prediction_volume.dtype}')
    
    #load the mask and calculate an iou
    if mskvol != 'none':
        mask_volume = sitk.GetArrayFromImage(sitk.ReadImage(mskvol))
        mask_volume = mask_volume.astype(np.uint8)

        if mask_prediction:
            prediction_volume *= mask_volume > 0
            labels = np.unique(mask_volume)[2:]
        elif num_classes == 1:
            labels = np.unique(mask_volume)[1:]
        else:
            labels = np.unique(mask_volume)

        if model_class > -1 and mask_class > -1:
            pred_labels = [model_class]
            mask_labels = [mask_class]
        else:
            pred_labels = labels
            mask_labels = labels

        ious = []
        print(f'Evaluating mask class(es) {mask_labels}...')
        for pl, ml in zip(pred_labels, mask_labels):
            label_pred = prediction_volume == pl
            label_gt = mask_volume == ml
            intersect = (label_pred * label_gt).sum()
            union = label_pred.sum() + label_gt.sum() - intersect
            iou = (intersect + 1e-7) / (union + 1e-7)

            ious.append(iou)

        if len(mask_labels) > 1:
            ious = np.array(ious)
            string_ious = ','.join([f'{iou:.5f}' for iou in ious])
            print(f'Class IoUs {string_ious}')
            print(f'Mean IoU 3d {ious.mean():.5f}')
        else:
            print(f'IoU 3d {ious[0]:.5f}')
    
    if save_path != 'none':
        prediction_volume = sitk.GetImageFromArray(prediction_volume)
        prediction_volume.CopyInformation(orig_vol)
        #the label_vol should be good now, let's save the result and finish
        #set the directions to match
        sitk.WriteImage(prediction_volume, save_path)
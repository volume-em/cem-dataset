"""
Description:
------------

Fits a ResNet34 model to images that have manually been labeled as "informative" or "uninformative". It's assumed that 
images have been manually labeled using the corrector.py utilities running in a Jupyter notebook (see notebooks/labeling.ipynb).

The results of this script are the roc curve plot on a randomly chosen validation set of images, the
model state dict as a .pth file and the model's predictions on all the remaining unlabeled images.

Example usage:
--------------

python classify_nn.py {impaths_file} {savedir} --labels {label_file} --weights {weights_file}

For help with arguments:
------------------------

python classify_nn.py --help
"""

DEFAULT_WEIGHTS = "https://www.dropbox.com/s/2libiwgx0qdgxqv/patch_quality_classifier_nn.pth?raw=1"

import os, sys, cv2, argparse
import numpy as np
import dask.array as da
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet34
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(
        description='Classifies a set of images by fitting a random forest to an array of descriptive features'
    )
    parser.add_argument('impaths_file', type=str, metavar='impaths_file', 
                        help='Path to .npz dask array file containing patch filepaths (for example deduplicated_fpaths.npz)')
    parser.add_argument('savedir', type=str, metavar='savedir', 
                        help='Directory in which to save predictions')
    parser.add_argument('--labels', type=str, metavar='labels',
                        help='Optional, path to array file containing image labels (informative or uninformative)')
    parser.add_argument('--weights', type=str, metavar='weights',
                        help='Optional, path to nn weights file. The default is to download weights used in the paper.')
    
    #parse the arguments
    args = parser.parse_args()
    impaths_file = args.impaths_file
    savedir = args.savedir
    gt_labels = args.labels
    weights = args.weights
    
    #make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    #load the dask array
    impaths = da.from_npy_stack(impaths_file)

    #load the labels array (if there is one)
    if gt_labels is not None:
        gt_labels = np.load(gt_labels)
    else:
        gt_labels = np.array(len(impaths) * ['none'])
        
    #sanity check that the number of labels and impaths are the same
    assert(len(impaths) == len(gt_labels)), "Number of impaths and labels are different!"

    #it's expected that the gt_labels were generated within a Jupyter notebook by
    #using the the corrector.py labeling utilities 
    #in that case the labels are text with the possible options of "good", "bad", and "none"
    #those with the label "none" are considered the unlabeled set and we make predictions
    #about their labels using the random forest that we train on the labeled images
    good_indices = np.where(gt_labels == 'informative')[0]
    bad_indices = np.where(gt_labels == 'uninformative')[0]
    labeled_indices = np.concatenate([good_indices, bad_indices], axis=0)
    unlabeled_indices = np.setdiff1d(range(len(impaths)), labeled_indices)

    #create the test set
    tst_impaths = impaths[unlabeled_indices].compute()
    
    #set up evaluation transforms (assumes imagenet pretrained as default
    #in train_nn.py)
    imsize = 224
    normalize = Normalize() #default is imagenet normalization
    eval_tfs = Compose([
        Resize(imsize, imsize),
        normalize,
        ToTensorV2()
    ])

    #make a basic dataset class for loading and augmenting images
    #WITHOUT any labels
    class SimpleDataset(Dataset):
        def __init__(self, imfiles, tfs=None):
            super(SimpleDataset, self).__init__()
            self.imfiles = imfiles
            self.tfs = tfs
            
        def __len__(self):
            return len(self.imfiles)
        
        def __getitem__(self, idx):
            #load the image
            image = cv2.imread(self.imfiles[idx], 0)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #apply transforms
            if self.tfs is not None:
                image = self.tfs(image=image)['image']
                
            return {'image': image}
        
    #create datasets for the train, validation, and test sets
    tst_data = SimpleDataset(tst_impaths, eval_tfs)

    #create the test dataload
    test = DataLoader(tst_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    #create the resnet34 model
    model = resnet34()

    #modify the output layer to predict 1 class only
    model.fc = nn.Linear(in_features=512, out_features=1)
    
    #load the weights from file or from online
    if weights is not None:
        state_dict = torch.load(weights)
    else:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS)
        
    #load in the weights (strictly)
    msg = model.load_state_dict(state_dict)
    
    #move the model to a cuda device
    model = model.cuda()

    #faster training
    cudnn.benchmark = True
    
    #lastly run inference on the entire set of unlabeled images
    print(f'Running inference on test set...')
    tst_predictions = []
    for data in tqdm(test):
        with torch.no_grad():
            #load data onto gpu
            #then forward pass
            images = data['image'].cuda(non_blocking=True)

            output = model.eval()(images)
            pred = nn.Sigmoid()(output)
        
        tst_predictions.append(pred.detach().cpu().numpy())

    tst_predictions = np.concatenate(tst_predictions, axis=0)

    #create an array of labels that are all zeros and fill in the values from a combination
    #of the ground truth labels from training and validation sets and the predicted
    #labels for unlabeled indices
    #convert gt_labels from strings to integers
    predicted_labels = (gt_labels == 'informative').astype(np.uint8)
    predicted_labels[unlabeled_indices] = (tst_predictions[:, 0] > 0.5).astype(np.uint8)

    print(f'Saving predictions...')
    np.save(os.path.join(savedir, "nn_predictions.npy"), predicted_labels)
    
    print(f'Saving filepaths...')
    filtered_fpaths = da.from_array(impaths[predicted_labels == 1].compute())
    da.to_npy_stack(os.path.join(savedir, 'nn_filtered_fpaths.npz'), filtered_fpaths)

    print('Finished.')
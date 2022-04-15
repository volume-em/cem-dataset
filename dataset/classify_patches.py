"""
Description:
------------

Classifies EM images into "informative" or "uninformative".

Example usage:
--------------

python classify_nn.py {deduped_dir} {savedir} --labels {label_file} --weights {weights_file}

For help with arguments:
------------------------

python classify_nn.py --help
"""

DEFAULT_WEIGHTS = "https://zenodo.org/record/6458015/files/patch_quality_classifier_nn.pth?download=1"

import os, sys, cv2, argparse
import pickle
import numpy as np
from skimage import io
from glob import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet34
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classifies a set of images by fitting a random forest to an array of descriptive features'
    )
    parser.add_argument('dedupe_dir', type=str, help='Directory containing ')
    parser.add_argument('savedir', type=str)
    parser.add_argument('--weights', type=str, metavar='weights',
                        help='Optional, path to nn weights file. The default is to download weights used in the paper.')
    
    args = parser.parse_args()
    
    # parse the arguments
    dedupe_dir = args.dedupe_dir
    savedir = args.savedir
    weights = args.weights
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # list all pkl deduplicated files
    fpaths = glob(os.path.join(dedupe_dir, '*.pkl'))
    
    # set up evaluation transforms (assumes imagenet 
    # pretrained as default in train_nn.py)
    imsize = 224
    normalize = Normalize() #default is imagenet normalization
    eval_tfs = Compose([
        Resize(imsize, imsize),
        normalize,
        ToTensorV2()
    ])
    
    # create the resnet34 model
    model = resnet34()

    # modify the output layer to predict 1 class only
    model.fc = nn.Linear(in_features=512, out_features=1)
    
    # load the weights from file or from online
    # load the weights from file or from online
    if weights is not None:
        state_dict = torch.load(weights, map_location='cpu')
    else:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS)
        
    # load in the weights (strictly)
    msg = model.load_state_dict(state_dict)
    model = model.to(device)
    cudnn.benchmark = True

    # make a basic dataset class for loading and 
    # augmenting images WITHOUT any labels
    class SimpleDataset(Dataset):
        def __init__(self, image_dict, tfs=None):
            super(SimpleDataset, self).__init__()
            self.image_dict = image_dict
            self.tfs = tfs
            
        def __len__(self):
            return len(self.image_dict['names'])
        
        def __getitem__(self, idx):
            # load the image
            fname = self.image_dict['names'][idx]
            image = self.image_dict['patches'][idx]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # apply transforms
            if self.tfs is not None:
                image = self.tfs(image=image)['image']
                
            return {'fname': fname, 'image': image}
        
    for fp in tqdm(fpaths):
        dataset_name = os.path.basename(fp)
        if '-ROI-' in dataset_name:
            dataset_name = dataset_name.split('-ROI-')[0]
        else:
            dataset_name = dataset_name[:-len('.pkl')]
        
        dataset_savedir = os.path.join(savedir, dataset_name)
        if not os.path.exists(dataset_savedir):
            os.mkdir(dataset_savedir)
        else:
            continue

        # load the patches_dict
        with open(fp, mode='rb') as handle:
            patches_dict = pickle.load(handle)
            
        # create datasets for the train, validation, and test sets
        tst_data = SimpleDataset(patches_dict, eval_tfs)
        test = DataLoader(tst_data, batch_size=128, shuffle=False, 
                          pin_memory=True, num_workers=4)

        # lastly run inference on the entire set of unlabeled images
        tst_fnames = []
        tst_predictions = []
        for data in test:
            with torch.no_grad():
                # load data onto gpu then forward pass
                images = data['image'].to(device, non_blocking=True)
                output = model.eval()(images)
                predictions = nn.Sigmoid()(output)
                
            predictions = predictions.detach().cpu().numpy()
            tst_predictions.append(predictions)
            tst_fnames.append(data['fname'])
            
        tst_fnames = np.concatenate(tst_fnames, axis=0)
        tst_predictions = np.concatenate(tst_predictions, axis=0)
        tst_predictions = (tst_predictions[:, 0] > 0.5).astype(np.uint8)
        
        for ix, (fn, img) in enumerate(zip(patches_dict['names'], patches_dict['patches'])):
            if tst_predictions[ix] == 1:
                io.imsave(os.path.join(dataset_savedir, fn + '.tiff'), img, check_contrast=False)
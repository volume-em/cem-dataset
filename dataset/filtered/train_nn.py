"""
Description:
------------

Fits a ResNet34 model to images that have manually been labeled as "informative" or "uninformative". It's assumed that 
images have been manually labeled using the corrector.py utilities running in a Jupyter notebook (see notebooks/labeling.ipynb).

The results of this script are the roc curve plot on a randomly chosen validation set of images and the
model state dict as a .pth file.

Example usage:
--------------

python train_nn.py {impaths_file} {labels_fpath} {savedir}

For help with arguments:
------------------------

python train_nn.py --help
"""

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

from albumentations import (
    Compose, PadIfNeeded, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
    CropNonEmptyMaskIfExists, GaussNoise, RandomBrightnessContrast, RandomResizedCrop, Rotate, RandomCrop,
    GaussianBlur, CenterCrop, RandomGamma, ElasticTransform, Resize
)
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
    parser.add_argument('labels_fpath', type=str, metavar='labels_fpath', 
                        help='Path to array file containing image labels (good or bad)')
    parser.add_argument('savedir', type=str, metavar='savedir', 
                        help='Directory in which to save model and evaluation plots')
    
    #parse the arguments
    args = parser.parse_args()
    impaths_file = args.impaths_file
    labels_fpath = args.labels_fpath
    savedir = args.savedir
    
    #make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    #load the dask array
    impaths = da.from_npy_stack(impaths_file)

    #load the labels array
    gt_labels = np.load(labels_fpath)

    #sanity check that the number of labels and impaths are the same
    assert(len(impaths) == len(gt_labels)), "Number of impaths and labels are different!"

    #it's expected that the gt_labels were generated within a Jupyter notebook by
    #using the the corrector.py labeling utilities 
    #in that case the labels are text with the possible options of "informative", "uninformative", and "none"
    #those with the label "none" are considered the unlabeled set and we make predictions
    #about their labels using the random forest that we train on the labeled images
    good_indices = np.where(gt_labels == 'informative')[0]
    bad_indices = np.where(gt_labels == 'uninformative')[0]
    labeled_indices = np.concatenate([good_indices, bad_indices], axis=0)

    #setting the random seed ensures that we're always comparing results against the same
    #validation dataset (otherwise hyperparameter tuning results would be indecipherable)
    #resetting the seed to something random ensures that there aren't issues with the
    #random augmentations
    np.random.seed(1227)
    trn_indices, val_indices = train_test_split(labeled_indices, test_size=0.15)
    np.random.seed(None)

    #convert the labels from text to integers (0 = "uninformative", 1= "informative")
    labels = np.zeros((len(impaths), ))
    labels[good_indices] = 1

    #separate train and validation sets
    trn_impaths = impaths[trn_indices].compute()
    trn_labels = labels[trn_indices]

    val_impaths = impaths[val_indices].compute()
    val_labels = labels[val_indices]

    #augmentations are carefully chosen such that the amount of distortion would not
    #transform an otherwise "informative" patch into an "uninformative" patch (for example, by making it low contrast)
    imsize = 224
    normalize = Normalize() #default is imagenet normalization
    tfs = Compose([
        Resize(imsize, imsize),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        GaussNoise(var_limit=(40, 100.0), p=0.5),
        GaussianBlur(blur_limit=5, p=0.5),
        HorizontalFlip(),
        VerticalFlip(),
        normalize,
        ToTensorV2()
    ])

    eval_tfs = Compose([
        Resize(imsize, imsize),
        normalize,
        ToTensorV2()
    ])

    #make a basic dataset class for loading and augmenting images
    class SimpleDataset(Dataset):
        def __init__(self, imfiles, labels, tfs=None):
            super(SimpleDataset, self).__init__()
            self.imfiles = imfiles
            self.labels = labels
            self.tfs = tfs
            
        def __len__(self):
            return len(self.imfiles)
        
        def __getitem__(self, idx):
            #load the image
            image = cv2.imread(self.imfiles[idx], 0)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #load the label
            label = self.labels[idx]
            
            #apply transforms
            if self.tfs is not None:
                image = self.tfs(image=image)['image']
                
            return {'image': image, 'label': label}
        
    #create datasets for the train and validation sets
    trn_data = SimpleDataset(trn_impaths, trn_labels, tfs)
    val_data = SimpleDataset(val_impaths, val_labels, eval_tfs)

    #create dataloaders
    bsz = 64
    train = DataLoader(trn_data, batch_size=bsz, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    valid = DataLoader(val_data, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=4)

    #create the model initialized with ImageNet weights
    model = resnet34(pretrained=True)

    #freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    #modify the output layer to predict 1 class only
    model.fc = nn.Linear(in_features=512, out_features=1)

    #move the model to a cuda device
    model = model.cuda()

    #unfreeze all parameters below the given finetune layer
    finetune_layer = 'layer4'
    backbone_groups = [mod[1] for mod in model.named_children()]
    if finetune_layer != 'none':
        layer_index = {'all': 0, 'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
        start_layer = layer_index[finetune_layer]

        #always finetune from the start layer to the last layer in the resnet
        for group in backbone_groups[start_layer:]:
            for param in group.parameters():
                param.requires_grad = True
                    

    #print the number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Using model with {params} trainable parameters!')

    #define loss and optiimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    #faster training
    cudnn.benchmark = True

    #basic accuracy calculation
    def accuracy(output, labels):
        #squeeze both
        output = output.squeeze()
        labels = labels.squeeze() > 0
        
        #sigmoid output
        output = nn.Sigmoid()(output) > 0.5
        
        #measure correct
        correct = torch.sum(output == labels).float()
        return (correct / len(labels)).item()


    #runs model training and validation loops for 20 epochs
    for epoch in range(20):
        rl = 0
        ra = 0
        for data in tqdm(train):
            #load data onto gpu
            images = data['image'].cuda(non_blocking=True)
            labels = data['label'].cuda(non_blocking=True)

            #zero grad
            optimizer.zero_grad()

            #forward
            output = model.train()(images)
            loss = criterion(output, labels.unsqueeze(1))

            #backward
            loss.backward()

            #step
            optimizer.step()

            rl += loss.item()
            ra += accuracy(output, labels)

        print(f'Epoch {epoch + 1}, Loss {rl / len(train)}, Accuracy {ra / len(train)}')
        
        rl = 0
        ra = 0
        for data in valid:
            #load data onto gpu
            images = data['image'].cuda(non_blocking=True)
            labels = data['label'].cuda(non_blocking=True)
            
            #forward
            output = model.eval()(images)
            loss = criterion(output, labels.unsqueeze(1))
            rl += loss.item()
            ra += accuracy(output, labels)

        print(f'Val Loss {rl / len(valid)}, Accuracy {ra / len(valid)}')

    #save the model
    torch.save(model.state_dict(), os.path.join(savedir, 'patch_quality_classifier_nn.pth'))
    print(f'Model finished training, weights saved to {savedir}')

    #run more extensive validation and print results
    print(f'Evaluating model predictions...')
    val_predictions = []
    for data in tqdm(valid):
        #load data onto gpu
        images = data['image'].cuda(non_blocking=True)

        #forward
        output = model.eval()(images)
        pred = nn.Sigmoid()(output)
        val_predictions.append(pred.detach().cpu().numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)

    tn, fp, fn, tp = confusion_matrix(val_labels, val_predictions > 0.5).ravel()
    acc = accuracy_score(val_labels, val_predictions > 0.5)

    print(f'Total validation images: {len(val_data)}')
    print(f'True Positives: {tp}')
    print(f'True Negatives: {tn}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'Accuracy: {acc}')

    #plot roc curve and save it
    fpr_nn, tpr_nn, _ = roc_curve(val_labels, val_predictions)
    plt.plot(fpr_nn, tpr_nn, linewidth=8, label=f'ConvNet (AUC = {roc_auc_score(val_labels, val_predictions):.3f})')
    plt.xlabel('False positive rate', labelpad=16, fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.ylabel('True positive rate', labelpad=16, fontsize=18, fontweight="bold")
    plt.yticks(fontsize=14)
    plt.title('NN Patch Quality Classifier ROC Curve', fontdict={'fontsize': 22, 'fontweight': "bold"}, pad=24)
    plt.tight_layout()
    plt.legend(loc='best', fontsize=18)
    plt.savefig(os.path.join(savedir, "patch_quality_nn_roc_curve.png"))
    
    print('Finished!')
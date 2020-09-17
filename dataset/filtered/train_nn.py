import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from albumentations import (
    Compose, PadIfNeeded, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
    CropNonEmptyMaskIfExists, GaussNoise, RandomBrightnessContrast, RandomResizedCrop, Rotate, RandomCrop,
    GaussianBlur, CenterCrop, RandomGamma, ElasticTransform
)

from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet34

from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

normalize = Normalize()

crop_size = 224
tfs = Compose([
    PadIfNeeded(min_height=crop_size, min_width=crop_size),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    GaussNoise(var_limit=(40, 100.0), p=0.5),
    GaussianBlur(blur_limit=5, p=0.5),
    HorizontalFlip(),
    VerticalFlip(),
    normalize,
    ToTensorV2()
])

eval_tfs = Compose([
    PadIfNeeded(min_height=crop_size, min_width=crop_size),
    normalize,
    ToTensorV2()
])


imfiles = np.load('/data/IASEM/conradrw/data/images224_fpaths_qsf.npy')
gt_labels = np.load('/data/IASEM/conradrw/data/images224_fpaths_qsf_rf_gt.npy')

good_indices = np.where(gt_labels == 'good')[0]
bad_indices = np.where(gt_labels == 'bad')[0]
labeled_indices = np.concatenate([good_indices, bad_indices], axis=0)
unlabeled_indices = np.setdiff1d(range(len(features)), labeled_indices)

from sklearn.model_selection import train_test_split
np.random.seed(1227)
trn_indices, val_indices = train_test_split(labeled_indices, test_size=0.2)
np.random.seed(None)

labels = np.zeros((len(imfiles), ))
labels[good_indices] = 1
trn_imfiles = imfiles[trn_indices]
trn_labels = labels[trn_indices]
val_imfiles = imfiles[val_indices]
val_labels = labels[val_indices]
tst_imfiles = imfiles[unlabeled_indices]
tst_labels = labels[unlabeled_indices]

import cv2

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
        image = imread(self.imfiles[idx])
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        #load the label
        label = self.labels[idx]
        
        #apply transforms
        if self.tfs is not None:
            image = self.tfs(image=image)['image']
            
        return {'image': image, 'label': label}
    
trn_data = SimpleDataset(trn_imfiles, trn_labels, tfs)
val_data = SimpleDataset(val_imfiles, val_labels, eval_tfs)
tst_data = SimpleDataset(tst_imfiles, tst_labels, eval_tfs)

bsz = 64
train = DataLoader(trn_data, batch_size=bsz, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
valid = DataLoader(val_data, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=4)
test = DataLoader(tst_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

model = resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Linear(in_features=512, out_features=1)

state = torch.load('/data/IASEM/conradrw/data/nn_filtering.pth', map_location='cpu')
model.load_state_dict(state)

model = model.cuda()

#freeze all backbone layers to start and only open
#them when specified
#for param in model.parameters():
#    param.requires_grad = False

finetune_layer = 'layer4'
backbone_groups = [mod[1] for mod in model.named_children()]
if finetune_layer != 'none':
    layer_index = {'all': 0, 'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
    start_layer = layer_index[finetune_layer]

    #always finetune from the start layer to the last layer in the resnet
    for group in backbone_groups[start_layer:]:
        for param in group.parameters():
            param.requires_grad = True
                
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'Using model with {params} trainable parameters!')

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

cudnn.benchmark = True

def accuracy(output, labels):
    #squeeze both
    output = output.squeeze()
    labels = labels.squeeze() > 0
    
    #sigmoid output
    output = nn.Sigmoid()(output) > 0.5
    
    #measure correct
    correct = torch.sum(output == labels).float()
    return (correct / len(labels)).item()

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
    for data in tqdm(valid):
        #load data onto gpu
        images = data['image'].cuda(non_blocking=True)
        labels = data['label'].cuda(non_blocking=True)
        
        #forward
        output = model.eval()(images)
        loss = criterion(output, labels.unsqueeze(1))
        rl += loss.item()
        ra += accuracy(output, labels)

    print(f'Val Loss {rl / len(valid)}, Accuracy {ra / len(valid)}')
    
predictions = []
for data in tqdm(train):
    #load data onto gpu
    images = data['image'].cuda(non_blocking=True)

    #forward
    output = model.eval()(images)
    pred = nn.Sigmoid()(output)
    predictions.append(pred.detach().cpu().numpy())
    
all_predictions = np.zeros((len(imfiles),)).astype(np.uint8)
all_predictions[trn_indices] = trn_labels.astype(np.uint8)
all_predictions[val_indices] = val_labels.astype(np.uint8)
all_predictions[unlabeled_indices] = (tst_predictions[:, 0] > 0.5).astype(np.uint8)
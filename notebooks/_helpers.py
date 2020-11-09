import sys, os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from scipy.stats.mstats import pointbiserialr
from skimage.transform import resize
import torchvision.transforms as tf
from torchvision.models import resnet50

sys.path.append('../pretraining_moco/')
from dataset import GaussianBlur, GaussNoise
from builder import MoCo
from resnet import resnet50 as moco_resnet50

CELLEMNET_WEIGHTS_URL = "https://www.dropbox.com/s/bqw4h2x23v3cgup/cellemnet_filtered_moco_v2_200ep.pth.tar?raw=1"
IMAGENET_MOCO_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar"

__all__ = [
    'load_full_moco_cellemnet', 'load_moco_pretrained', 'DefaultDecoder', 'DataFolder', 
    'rescale', 'CorrelationData', 'restride_resnet', 'correlated_filters', 'mean_topk_map', 'binary_iou',
    'transform_local_trajectory', 'calculate_mean_firing_rates', 'GaussianBlur', 'GaussNoise'
]

#download the CellEMNet weights and load them
#into the MoCo resnet50 clss, it includes
#all MoCo parameters like the query and key encoders
def load_full_moco_cellemnet():
    #download the weights
    model_state = torch.hub.load_state_dict_from_url(CELLEMNET_WEIGHTS_URL)

    #strip the module prefix used when training models
    #in parallel
    state_dict = model_state['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        new_key = k.split('module.')[-1]
        state_dict[new_key] = state_dict[k]
        del state_dict[k]

    #extract the mean and std pixel values
    norms = model_state['norms']
    
    #load the model and return along with the norms
    model = MoCo(moco_resnet50, 128, 65536, 0.999, 0.2, True)
    msg = model.load_state_dict(state_dict)
    
    return model, norms

def load_moco_pretrained(dataset='cellemnet'):
    #download the weights
    if dataset == 'cellemnet':
        model_state = torch.hub.load_state_dict_from_url(CELLEMNET_WEIGHTS_URL)
        model = moco_resnet50()
        #extract the mean and std pixel values
        norms = model_state['norms']
    else:
        model_state = torch.hub.load_state_dict_from_url(IMAGENET_MOCO_WEIGHTS_URL)
        model = resnet50()
        norms = None #imagenet default norms instead
        
    
    state_dict = model_state['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #remove the prefix such that the names match the resnet50 model
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
        
    #load the model and return along with the norms
    msg = model.load_state_dict(state_dict, strict=False)
    
    return model, norms

class DefaultDecoder(nn.Module):
    def __init__(self, encoder):
        super(DefaultDecoder, self).__init__()
        self.encoder = encoder
        
    def compare_encodings(self, query, key):
        #normalize both of them
        query = nn.functional.normalize(query, dim=1)
        key = nn.functional.normalize(key, dim=1)
        
        #because of how the occlusion function works
        #we need to make copies of the key
        key = key.repeat(query.size(0), 1)
        
        #return the comparisons
        return torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        
    def forward(self, image1):
        pass
    
def rescale(tensor):
    #function to rescale a numpy image to fall in [0-1]
    tensor = np.copy(tensor)
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor

class DataFolder(Dataset):
    """
    Extremely simply dataset class that loads images from a directory
    and applies two sets of random transforms
    """
    
    def __init__(self, imdir, tfs):
        super(DataFolder, self).__init__()
        self.imdir = imdir
        self.fnames = next(os.walk(self.imdir))[2]
        print(f'Found {len(self.fnames)} images in {imdir}')
        self.tfs = tfs
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        #open the image with il
        image = Image.open(os.path.join(self.imdir, self.fnames[idx]))
            
        #return the differently transformed images
        return self.tfs(image), self.tfs(image)

class CorrelationData(Dataset):
    """
    Simple dataset class for loading images and masks for the heatmap
    correlation testing. Assumes transforms are through albumentations
    """
    
    def __init__(self, data_dir, tfs=None, gray_channels=3):
        super(CorrelationData, self).__init__()
        self.data_dir = data_dir
        self.impath = os.path.join(data_dir, 'images')
        self.mskpath = os.path.join(data_dir, 'masks')
        
        #get filenames and remove any hidden files 
        #that might be in the directory
        self.fnames = next(os.walk(self.impath))[2]
        self.fnames = [fn for fn in self.fnames if fn[0] != '.']
        print(f'Found {len(self.fnames)} images in {self.impath}')
        
        self.tfs = tfs
        self.gray_channels = gray_channels
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        #load the image and mask files
        f = self.fnames[idx]
        image = cv2.imread(os.path.join(self.impath, f), 0)
        
        #binarize the mask
        mask = (cv2.imread(os.path.join(self.mskpath, f), 0) > 0).astype(np.uint8)
        
        #albumentations expects a channel dimension
        #there can be three or one
        if self.gray_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[..., None]
        
        #make the output dict and store the original
        #image shape before augmentation
        output = {'fname': f, 'image': image, 'mask': mask}
        
        #apply transforms, assumes albumentations
        if self.tfs is not None:
            transformed = self.tfs(**output)
            output['image'] = transformed['image']
            output['mask'] = transformed['mask']
                
        return output
    
#changes the resolution of the resnet output
#feature maps
def restride_resnet(resnet, downsample_factor=4):
    #factor determines how much larger to
    #make the feature maps, accepted
    #accepted values are 2, 4, 8, 16, 32
    assert(downsample_factor in [2, 4, 8, 16, 32]), \
    "Invalid factor given, must be one of 2, 4, 8, 16, or 32"
    
    #if the factor is 42then, we only remove the downsampling
    #in layer 4 of the resnet, if 4, then layers 3&4 and if
    #8 then layers 2&3&4
    if downsample_factor <= 16:
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
    
    if downsample_factor <= 8:
        resnet.layer3[0].conv2.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        
    if downsample_factor <= 4:
        resnet.layer2[0].conv2.stride = (1, 1)
        resnet.layer2[0].downsample[0].stride = (1, 1)
        
    if downsample_factor <= 2:
        resnet.conv1.stride = (1, 1)
        
    return resnet

#this function measures the correlation
#between locations with high filter responses and 
#a ground truth labelmap (mask)
def correlated_filters(model, image, mask):
    #image is expected to be a tensor of
    #([1 or 3], H, W) on the same device
    #as the model
    #mask should be a tensor of (H, W)
    with torch.no_grad():
        #run the image through the model
        layer4_features = model.eval()(image.unsqueeze(0)).detach()
        
        #upsample to match image size
        layer4_features = F.interpolate(layer4_features, image.size()[-2:], mode='bilinear', align_corners=True)
        
        #convert to numpy
        layer4_features = layer4_features.squeeze().cpu().numpy() #(2048, 224, 224)
        
    #loop over each of the features and calculate correlation with the mask
    corrs = []
    for fm in layer4_features:
        pbr = pointbiserialr(mask.numpy().ravel().astype(np.bool), fm.ravel())[0]
        corrs.append(pbr)
        
    return np.array(corrs)    

#this function takes the mean of the top k filters
#and optionally rescales the results to fall in the
#range 0-1
def mean_topk_map(model, image, topk_indices, rescale=True):
    with torch.no_grad():
        #run the image through the model
        layer4_features = model.eval()(image.unsqueeze(0))
        
    #get the mean over the topk filters as numpy array
    mean_fmap = layer4_features[:, topk_indices].mean(dim=1).detach().cpu().squeeze().numpy()
    
    #resize to the original size and rescale from 0-1
    h, w = image.size()[-2:]
    mean_fmap = resize(mean_fmap, (h, w), order=1)
    if rescale:
        mean_fmap -= mean_fmap.min()
        mean_fmap /= mean_fmap.max()
        
    return mean_fmap

#simple iou calculator for binary
def binary_iou(pred, mask):
    intersect = np.logical_and(pred == 1, mask == 1).sum()
    union = np.logical_or(pred == 1, mask == 1).sum()
    return intersect / union

def transform_local_trajectory(model, dataset, base_aug_list, additive_augs, aug_insert_loc):
    #define a list for the trajectories
    local_trajectories = []
    
    #determine the model device for later
    device = next(model.parameters()).device
    
    for idx in range(len(dataset)):
        #define list for the trajectory on a single image
        trajectory = []
        #loop over all levels of distortion included
        #in the list of additive_augs
        for aug in additive_augs:
            #first create the new list of augmentations
            augs = base_aug_list[:aug_insert_loc] + [aug] + base_aug_list[aug_insert_loc:]
            
            #overwrite the datasets transforms with the composed augmentations
            dataset.tfs = tf.Compose(augs)
            
            #load the image
            aug_image, _ = dataset[idx]
            
            #move the image to the model device
            #and add the batch dimension
            aug_image = aug_image.to(device).unsqueeze(0)
            
            #calculate the response
            with torch.no_grad():
                response = model(aug_image)
                
            #response is a vector of shape (2048,)
            trajectory.append(response.detach().cpu().numpy().ravel())
            
        #add the trajectory over all the augmentation strengths
        #as 1 complete local trajectory for an image
        #stacked trajectory has shape (len(additive_augs), 2048)
        local_trajectories.append(np.stack(trajectory, axis=0))
        
    #return the local trajectories for all images
    #array of shape (len(dataset), len(additive_augs), 2048)
    return np.stack(local_trajectories, axis=0)

def calculate_mean_firing_rates(local_trajectories, firing_thresholds, trajectory_ref_index=0, near_max_percentile=0.9):
    #start by calculating the near maximum responses
    near_maximums = np.quantile(local_trajectories[:, trajectory_ref_index], near_max_percentile, axis=0)
    
    #loop over the tested augmentation strengths
    #and compare firing rates relative to
    #the undistorted reference image in the local_trajectory
    mean_firing_rates = []
    for i in range(local_trajectories.shape[1]):
        #loop over all of the features (neurons) and measure
        #their local firing rates (typically there are 2048)
        local_firing_rates = []
        for j in range(local_trajectories.shape[2]):
            #check that the near maximum value is over the firing threshold
            #if it's not then we ignore that neuron because it isn't selective
            #for anything in the images that we tested
            if near_maximums[j] <= firing_thresholds[j]:
                continue

            #get the indices of images that activate the given neuron
            #near maximally (notice that first dim of local_trajectories is over images
            #in the tested dataset)
            indices = np.where(local_trajectories[:, trajectory_ref_index, j] >= near_maximums[j])[0]

            #there should always be at least 1 index
            #and the near maximum should be greater than 0
            #just apply a sanity check to be sure
            assert(len(indices) > 0), f"No images activate neuron {j} near-maximally!"
            assert(near_maximums[j] > 0), f"Near maximum response for neuron {j} is 0!"

            if len(indices) > 0 and near_maximums[j] > 0:
                local_firing_rates.append((local_trajectories[indices][:, i, j] > firing_thresholds[j]).mean())
    
        mean_firing_rates.append(np.mean(local_firing_rates))
        
    #return the mean firing rates
    #size is equal to the number of tested
    #augmentation strengths
    return np.array(mean_firing_rates)
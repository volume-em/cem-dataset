import sys, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

sys.path.append('../pretraining_moco/')
from builder import MoCo
from resnet import resnet50 as moco_resnet50

WEIGHTS_URL = "https://www.dropbox.com/s/bqw4h2x23v3cgup/cellemnet_filtered_moco_v2_200ep.pth.tar?raw=1"

__all__ = ['load_pretrained_cellemnet', 'DefaultDecoder', 'DataFolder', 'rescale']

#download the CellEMNet weights and load them
#into the MoCo resnet50 model
def load_pretrained_cellemnet():
    #download the weights
    model_state = torch.hub.load_state_dict_from_url(WEIGHTS_URL)

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
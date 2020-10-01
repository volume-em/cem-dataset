import torch.hub

__all__ = ['moco_to_unet_prefixes', 'load_pretrained_state_for_unet']

cellemnet_moco_model_urls = {
    'resnet50': 'https://www.dropbox.com/s/bqw4h2x23v3cgup/cellemnet_filtered_moco_v2_200ep.pth.tar?raw=1'
}

imagenet_moco_model_urls = {
    'resnet50': 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
}

def moco_to_unet_prefixes(state_dict):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #for unet, we need to remove module.encoder_q. from the prefix
            #and add encoder instead
            state_dict['encoder.' + k[len("module.encoder_q."):]] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
        
    return state_dict


def load_pretrained_state_for_unet(model_name='resnet50', pretraining='cellemnet_mocov2'):
    #validate the pretraining dataset name
    if pretraining not in ['cellemnet_mocov2', 'imagenet_mocov2']:
        raise Exception(f'Pretraining must be either cellemnet_mocov2 or imagenet_mocov2, got {pretraining}')
    
    #get the url
    if pretraining == 'cellemnet_mocov2':
        url = cellemnet_moco_model_urls[model_name]
    else:
        url = imagenet_moco_model_urls[model_name]
    
    #download and save the weights to TORCH_HOME; after initial download, weight are
    #loaded from that path instead
    model_state = torch.hub.load_state_dict_from_url(url)
    
    # rename moco pre-trained keys
    state_dict = model_state['state_dict']
    state_dict = moco_to_unet_prefixes(state_dict)
        
    #return both the model state_dict and the norms used during training
    #if there are no norms, then we return None and will assume ImageNet
    if 'norms' in model_state:
        norms = model_state['norms']
    else:
        norms = None
    
    return state_dict, norms
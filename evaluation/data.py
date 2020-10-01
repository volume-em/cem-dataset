import cv2, os
from torch.utils.data import Dataset
from albumentations import DualTransform
from albumentations.augmentations import functional as AF

class SegmentationData(Dataset):
    """
    A relatively generic Pytorch dataset that loads image and mask files and
    applies augmentations (from albumentations).
    
    Arguments:
    ----------
    data_dir: String. A directory containing a subdirectory called "images" and another
    called "masks". Corresponding images and masks within the directories must have
    the same file names.
    
    tfs: albumentations.Compose. A set of transformations to apply to the image
    and mask data.
    
    gray_channels: Integer, choice of [1, 3]. Determines the number of channels in
    the grayscale image. Note that 3 channels are required for using ImageNet
    pretrained weights.
    
    segmentation_classes: Integer, the number of segmentation classes expected in
    masks. Default, 1 (binary segmentation).
    
    Example Usage:
    --------------
    from data import SegmentationData
    from albumentations import Compose, Normalize
    from albumentations.pytorch import ToTensorV2
    from matplotlib import pyplot as plt
    
    tfs = Compose([Normalize(), ToTensorV2()])
    dataset = SegmentationData(file_path, tfs, gray_channels=3)
    data = dataset[0]
    plt.imshow(data['image'][0], cmap='gray')
    plt.imshow(data['mask'], alpha=0.3)
    
    """
    
    def __init__(self, data_dir, tfs=None, gray_channels=3, segmentation_classes=1):
        super(SegmentationData, self).__init__()
        self.data_dir = data_dir
        self.impath = os.path.join(data_dir, 'images')
        self.mskpath = os.path.join(data_dir, 'masks')
        
        self.fnames = next(os.walk(self.impath))[2]
        print(f'Found {len(self.fnames)} images in {self.impath}')
        
        self.tfs = tfs
        self.gray_channels = gray_channels
        self.segmentation_classes = segmentation_classes
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        #load the image and mask files
        f = self.fnames[idx]
        image = cv2.imread(os.path.join(self.impath, f), 0)
        mask = cv2.imread(os.path.join(self.mskpath, f), 0)
        
        #albumentations expects a channel dimension
        #there can be three or one
        if self.gray_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[..., None]
        
        output = {'image': image, 'mask': mask}
        
        #apply transforms, assumes albumentations
        if self.tfs is not None:
            transformed = self.tfs(**output)
            output['image'] = transformed['image']
            output['mask'] = transformed['mask']
            
        #if there is more than 1 segmentation class we
        #need to set the mask to long type, otherwise
        #it should be float
        if self.segmentation_classes > 1:
            output['mask'] = output['mask'].long()
        else:
            #add a channel dimension
            output['mask'] = output['mask'].unsqueeze(0).float()
                
        return output
        
class FactorResize(DualTransform):

    def __init__(self, resize_factor, always_apply=False, p=1.0):
        super(FactorResize, self).__init__(always_apply, p)
        self.rf = resize_factor

    def apply(self, img, **params):
        h, w = img.shape[:2]
        nh = int(h / self.rf) * self.rf
        nw = int(w / self.rf) * self.rf
        
        #set the interpolator and return
        interpolation = cv2.INTER_NEAREST
        return AF.resize(img, nh, nw, interpolation)
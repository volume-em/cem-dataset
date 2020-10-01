import torch
import torch.nn as nn
from copy import deepcopy
from matplotlib import pyplot as plt

import numpy as np
import skimage.measure as measure

class AverageMeter:
    """Computes average values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count

class EMAMeter:
    """Computes and stores an exponential moving average and current value"""
    def __init__(self, momentum=0.98):
        self.mom = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = (self.sum * self.mom) + (val * (1 - self.mom))
        self.count += 1
        self.avg = self.sum / (1 - self.mom ** (self.count))

class IoU:
    """
    Computes IoU metric for batch of predictions and masks.
    
    Arguments:
    logits: If logits is True, then a sigmoid activation is applied to output
    before additional calculations. If False, then it is assumed that sigmoid
    activation has already been applied previously.
    
    Return:
    Calculate method adds calculated IoU value to a running average for epoch end
    evaluation.
    
    Usage:
    Case 1: Training loop
    
    iou = IoU()
    #NOTE: output.size == target.size
    for batch in epoch:
        ...
        iou.calculate(output, target)
        ...
    iou.epoch_end()
    print(iou.history) #prints the average value of IoU for all batches in the epoch
    
    Case 2: Single image or batch
    
    iou = IoU()
    #NOTE: output.size == target.size
    iou.calculate(output, target)
    iou.epoch_end()
    print(iou.history) #prints the value of IoU for output and target
    
    """
    
    def __init__(self, meter):
        self.meter = meter
        
    def calculate(self, output, target):
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
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersect = torch.sum(output * target, dims)
        
        #compute the union, (N,)
        union = torch.sum(output + target, dims) - intersect
        
        #avoid division errors by adding a small epsilon
        #if intersect and union are zero, then iou is 1
        iou = (intersect + 1e-5) / (union + 1e-5)
        
        return iou
    
    def update(self, value):
        self.meter.update(value)
        
    def reset(self):
        self.meter.reset()
        
    def average(self):
        return self.meter.avg
        
class ComposeMetrics:
    """
    A class for composing metrics together.
    
    Arguments:
    metrics_dict: A dictionary in which each key is the name of a metric and
    each value is a subclass of Metric with a calculate method that accepts
    model predictions and ground truth labels.
    
    Example:
    metrics_dict = {'IoU': IoU(), 'Dice coeff': Dice()}
    metrics = Compose(metrics_dict)
    #output.size == target.size
    metrics.evaluate(output, target)
    
    #to print value of IoU only
    #if close_epoch() is not called, history will be empty
    metrics.close_epoch()
    print(metrics.metrics['IoU'].history)
    
    #to print value of all metrics in metric_dict
    #if close_epoch() is not called, history will be empty
    metrics.close_epoch()
    print(metrics.print())
    """
    def __init__(self, metrics_dict, class_names=None, reset_on_print=True):
        self.metrics_dict = metrics_dict
        self.class_names = class_names
        self.reset_on_print = reset_on_print
        
    def evaluate(self, output, target):
        #calculate all the metrics in the dict
        for metric in self.metrics_dict.values():
            value = metric.calculate(output, target)
            metric.update(value)
            
    def print(self):
        names = []
        values = []
        for name, metric in self.metrics_dict.items():
            avg_values = metric.average().cpu()
            #we expect metric to be a tensor of size (n_classes,)
            #we want to print the corresponding class names if given
            if self.class_names is None:
                self.class_names = [f'class_{i}' for i in range(len(avg_values))]
            
            for class_name, val in zip(self.class_names, avg_values):
                names.append(f'{class_name}_{name}')
                values.append(val.item())
                
            if self.reset_on_print:
                metric.reset()
        
        for name, value in zip(names, values):
            print(f'{name}: {value:.3f}')
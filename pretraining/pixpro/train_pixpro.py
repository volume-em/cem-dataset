"""
Copied and modified from:
https://github.com/facebookresearch/moco/blob/master/main_moco.py

Which is really copied and modified from the source of all distributed
training scripts:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
import builtins
import math
import os, sys
import random
import shutil
import time
import warnings
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from LARC import LARC

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import ContrastData, Grayscale
from builder import PixPro, ConsistencyLoss

sys.path.append('/home/conradrw/nbs/cellemnet/pretraining/')
from resnet import resnet50

import mlflow

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch PixPro Training')
    parser.add_argument('config', help='Path to .yaml training config file')

    return vars(parser.parse_args())

def main():
    args = parse_args()
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if not os.path.isdir(config['model_dir']):
        os.mkdir(config['model_dir'])

    config['config_file'] = args['config']

    #world size is the number of processes that will run
    if config['dist_url'] == "env://" and config['world_size'] == -1:
        config['world_size'] = int(os.environ["WORLD_SIZE"])

    config['distributed'] = config['world_size'] > 1 or config['multiprocessing_distributed']

    ngpus_per_node = torch.cuda.device_count()
    config['ngpus_per_node'] = ngpus_per_node
    if config['multiprocessing_distributed']:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config['world_size'] = ngpus_per_node * config['world_size']
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config['gpu'], ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    config['gpu'] = gpu
    
    # suppress printing if not master process
    if config['multiprocessing_distributed'] and config['gpu'] != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if config['gpu'] is not None:
        print("Use GPU: {} for training".format(config['gpu']))

    if config['distributed']:
        if config['dist_url'] == "env://" and config['rank'] == -1:
            config['rank'] = int(os.environ["RANK"])
        if config['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config['rank'] = config['rank'] * ngpus_per_node + gpu
            
        dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'],
                                world_size=config['world_size'], rank=config['rank'])

    print("=> creating model '{}'".format(config['arch']))
    
    model = PixPro(
        resnet50,
        config['pixpro_mom'], config['ppm_layers'], config['ppm_gamma']
    )
    
    if config['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # Turn on SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if config['gpu'] is not None:
            torch.cuda.set_device(config['gpu'])
            model.cuda(config['gpu'])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config['batch_size'] = int(config['batch_size'] / ngpus_per_node)
            config['workers'] = int((config['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config['gpu'] is not None:
        torch.cuda.set_device(config['gpu'])
        model = model.cuda(config['gpu'])
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    #define loss criterion and optimizer
    criterion = ConsistencyLoss(distance_thr=config['pixpro_t']).cuda(config['gpu'])

    optimizer = configure_optimizer(model, config)

    # optionally resume from a checkpoint
    if config['resume']:
        if os.path.isfile(config['resume']):
            print("=> loading checkpoint '{}'".format(config['resume']))
            if config['gpu'] is None:
                checkpoint = torch.load(config['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config['gpu'])
                checkpoint = torch.load(config['resume'], map_location=loc)
            config['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config['resume'], checkpoint['epoch']))
        else:
            config['start_epoch'] = 0
            print("=> no checkpoint found at '{}'".format(config['resume']))

    cudnn.benchmark = True
    
    norms = config['norms']
    mean_pixel = norms['mean']
    std_pixel = norms['std']
    normalize = A.Normalize(mean=[mean_pixel], std=[std_pixel])

    #physical space only
    space_tfs = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
        Grayscale(3),
        A.HorizontalFlip(),
        A.VerticalFlip()
    ], additional_targets={'grid_y': 'image', 'grid_x': 'image'})

    #could work for both views
    view1_color_tfs = A.Compose([
        A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
        Grayscale(1),
        A.GaussianBlur(blur_limit=23, sigma_limit=(0.1, 2.0), p=1.0),
        normalize,
        ToTensorV2()
    ])

    #technically optional, but used in the BYOL paper
    view2_color_tfs = A.Compose([
        A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
        Grayscale(1),
        A.GaussianBlur(blur_limit=23, sigma_limit=(0.1, 2.0), p=0.1),
        A.GaussNoise(p=0.2),
        normalize,
        ToTensorV2()
    ])

    train_dataset = ContrastData(
        config['data_dir'], space_tfs, view1_color_tfs, view2_color_tfs
    )

    if config['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # encoder momentum is updated by STEP and not EPOCH
    config['train_steps'] = config['epochs'] * len(train_loader)
    config['current_step'] = config['start_epoch'] * len(train_loader)

    if config['fp16']:
        scaler = GradScaler()
    else:
        scaler = None
        
    #log parameters, if needed:
    if config['logging'] and (config['multiprocessing_distributed'] 
                              and config['rank'] % ngpus_per_node == 0):
        
        #end any old runs
        mlflow.end_run()
        mlflow.set_experiment(config['experiment_name'])
        mlflow.log_artifact(config['config_file'])

        #we don't want to add everything in the config
        #to mlflow parameters, we'll just add the most
        #likely to change parameters
        mlflow.log_param('data_dir', config['data_dir'])
        mlflow.log_param('architecture', config['arch'])
        mlflow.log_param('epochs', config['epochs'])
        mlflow.log_param('batch_size', config['batch_size'])
        mlflow.log_param('learning_rate', config['lr'])
        mlflow.log_param('pixpro_mom', config['pixpro_mom'])
        mlflow.log_param('ppm_layers', config['ppm_layers'])
        mlflow.log_param('ppm_gamma', config['ppm_gamma'])
        mlflow.log_param('pixpro_t', config['pixpro_t'])

    for epoch in range(config['start_epoch'], config['epochs']):
        if config['distributed']:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, epoch, config)

        if not config['multiprocessing_distributed'] or (config['multiprocessing_distributed']
                and config['rank'] % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config['arch'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'norms': [mean_pixel, std_pixel],
            }, is_best=False, filename=os.path.join(config['model_dir'], 'current.pth.tar'))
            
            #save checkpoint every save_freq epochs
            if (epoch + 1) % config['save_freq'] == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': config['arch'],
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'norms': [mean_pixel, std_pixel],
                }, is_best=False, filename=os.path.join(config['model_dir'] + 'checkpoint_{:04d}.pth.tar'.format(epoch + 1)))

def train(train_loader, model, criterion, optimizer, scaler, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        view1 = batch['view1']
        view1_grid = batch['view1_grid']
        view2 = batch['view2']
        view2_grid = batch['view2_grid']

        # measure data loading time
        data_time.update(time.time() - end)

        if config['gpu'] is not None:
            view1 = view1.cuda(config['gpu'], non_blocking=True)
            view1_grid = view1_grid.cuda(config['gpu'], non_blocking=True)
            view2 = view2.cuda(config['gpu'], non_blocking=True)
            view2_grid = view2_grid.cuda(config['gpu'], non_blocking=True)

        optimizer.zero_grad()

        # compute output and loss
        if config['fp16']:
            with autocast():
                output = model(view1, view2, view1_grid, view2_grid)
                loss = criterion(*output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(view1, view2, view1_grid, view2_grid)
            loss = criterion(*output)
            loss.backward()
            optimizer.step()

        # avg loss from batch size
        losses.update(loss.item(), view1.size(0))

        # update current step and encoder momentum
        config['current_step'] += 1
        adjust_encoder_momentum(model, config)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            progress.display(i)
            
        if config['rank'] % config['ngpus_per_node'] == 0:
            # store metrics to mlflow
            mlflow.log_metric('sim_loss', losses.avg, step=epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def configure_optimizer(model, config):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """
    decay = set()
    no_decay = set()

    blacklist = (nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = '%s.%s' % (mn, pn) if mn else pn

            if full_name.endswith('bias'):
                no_decay.add(full_name)
            elif full_name.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert(len(inter_params) == 0), "Overlapping decay and no decay"
    assert(len(param_dict.keys() - union_params) == 0), "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    #the adapt_lr key tells LARS not to adapt the lr (see 'LARC.py')
    param_groups = [
        {"params": decay_params, "weight_decay": config['weight_decay'], "adapt_lr": True},
        {"params": no_decay_params, "weight_decay": 0., "adapt_lr": False}
    ]

    base_optimizer = torch.optim.SGD(
        param_groups, lr=config['lr'], momentum=config['momentum']
    )

    #LARC without clipping == LARS
    #lower trust_coefficient to match SimCLR and BYOL
    #(too high of a trust_coefficient leads to NaN losses!)
    optimizer = LARC(optimizer=base_optimizer, trust_coefficient=1e-3)

    return optimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_encoder_momentum(model, config):
    base_mom = config['pixpro_mom']
    new_mom = 1 - (1 - base_mom) * (math.cos(math.pi * config['current_step'] / config['train_steps']) + 1) / 2
    model.module.momentum = new_mom

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config['lr']
    #cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / config['epochs']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

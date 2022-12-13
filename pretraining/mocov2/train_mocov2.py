"""

Copied with modification from https://github.com/facebookresearch/moco/blob/master/main_moco.py
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

Modifications:
--------------

1. Converted argparse namespace to a .yaml config file
2. Converted from tensorboard logging to mlflow logging
3. Added GaussNoise and Rotations to augmentations
4. Modified content of saved checkpoints to include the
   mean and std pixel values used for training and the mlflow run id
   
"""

import argparse
import builtins
import math
import os
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
import torchvision.transforms as tf

import mocov2.builder as builder
from mocov2.dataset import EMData, GaussianBlur, GaussNoise
import resnet as models

import mlflow

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
    parser.add_argument('config', help='Path to .yaml training config file')

    return vars(parser.parse_args())

def main():
    args = parse_args()
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
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
    
    model = builder.MoCo(
        models.__dict[config['arch']],
        config['moco_dim'], config['moco_k'], config['moco_m'], config['moco_t'], config['mlp']
    )

    if config['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
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

    criterion = nn.CrossEntropyLoss().cuda(config['gpu'])

    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    
    #set the start_epoch, overwritten if resuming
    config['start_epoch'] = 0
    
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
            print("=> no checkpoint found at '{}'".format(config['resume']))

    cudnn.benchmark = True
    
    #get the mean and standard deviation pixels from config
    #and wrap them in lists for tf.Normalize to work
    norms = config['norms']
    mean_pixel = norms['mean']
    std_pixel = norms['std']
    normalize = tf.Normalize(mean=[mean_pixel], std=[std_pixel])

    #for now, these augmentations are hardcoded. torchvision
    #isn't as easy to work with as albumentations
    augmentation = tf.Compose([
        tf.Grayscale(3),
        tf.RandomApply([tf.RandomRotation(180)], p=0.5),
        tf.RandomResizedCrop(224, scale=(0.2, 1.)),
        tf.ColorJitter(0.4, 0.4, 0.4, 0.1),
        tf.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        tf.Grayscale(1),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.ToTensor(),
        GaussNoise(p=0.5),
        normalize
    ])

    train_dataset = EMData(config['data_path'], augmentation)

    if config['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    
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
        mlflow.log_param('data_path', config['data_path'])
        mlflow.log_param('architecture', config['arch'])
        mlflow.log_param('epochs', config['epochs'])
        mlflow.log_param('batch_size', config['batch_size'])
        mlflow.log_param('learning_rate', config['lr'])
        mlflow.log_param('moco_dim', config['moco_dim'])
        mlflow.log_param('moco_k', config['moco_k'])
        mlflow.log_param('moco_m', config['moco_m'])
        mlflow.log_param('moco_t', config['moco_t'])

    for epoch in range(config['start_epoch'], config['epochs']):
        if config['distributed']:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config)

        #only save checkpoints from the main process
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


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        x1, x2 = torch.split(images, [1, 1], dim=1)

        if config['gpu'] is not None:
            x1 = x1.cuda(config['gpu'], non_blocking=True)
            x2 = x2.cuda(config['gpu'], non_blocking=True)

        # compute output
        output, target = model(im_q=x1, im_k=x2)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x1.size(0))
        top1.update(acc1[0], x1.size(0))
        top5.update(acc5[0], x1.size(0))        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            progress.display(i)
            
            
    if config['rank'] % config['ngpus_per_node'] == 0:
        # store metrics to mlflow
        mlflow.log_metric('ins_loss', losses.avg, step=epoch)
        mlflow.log_metric('top1_prob', top1.avg.item(), step=epoch)
        mlflow.log_metric('top5_prob', top5.avg.item(), step=epoch)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config['lr']
    if config['cos']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config['epochs']))
    else:  # stepwise lr schedule
        for milestone in config['schedule']:
            lr *= 0.1 if epoch >= milestone else 1.
            
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

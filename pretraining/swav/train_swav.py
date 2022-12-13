# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import builtins
import math
import os
import yaml
import shutil
import time
import mlflow

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as tf

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import models as models

from sampler import DistributedWeightedSampler
from LARC import LARC
from utils import (
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode
)
from dataset import MultiCropDataset, RandomGaussianBlur, GaussNoise

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SWaV Training')
    parser.add_argument('config', help='Path to .yaml training config file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config = {**config, **vars(args)}
    
    # load config dictionary into args
    args = argparse.Namespace(**config)
    
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
        
    #world size is the number of processes that will run
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    # suppress printing if not master process
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu

    # initialize distributed environment
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    # opt for random seeds for now
    #fix_random_seeds(args.seed)
    
    # set the image transforms
    mean, std = args.norms
    normalize = tf.Normalize(mean=[mean], std=[std])
    
    # list of transforms, one for each crop
    assert len(args.size_crops) == len(args.nmb_crops)
    assert len(args.min_scale_crops) == len(args.nmb_crops)
    assert len(args.max_scale_crops) == len(args.nmb_crops)
    transforms = []
    for i in range(len(args.size_crops)):
        crop_size = args.size_crops[i]
        min_scale = args.min_scale_crops[i]
        max_scale = args.max_scale_crops[i]
        num = args.nmb_crops[i]
        
        transforms.extend([tf.Compose([
            tf.Grayscale(3),
            tf.RandomApply([tf.RandomRotation(180)], p=0.5),
            tf.RandomResizedCrop(crop_size, scale=(min_scale, max_scale)),
            tf.ColorJitter(0.4, 0.4, 0.4, 0.1),
            RandomGaussianBlur(0.5, 0.1, 2.),
            tf.Grayscale(1),
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip(),
            tf.ToTensor(),
            GaussNoise(p=0.5),
            normalize
        ])] * num)

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        transforms,
        args.weight_gamma
    )
    
    if train_dataset.weights is None:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = DistributedWeightedSampler(train_dataset, train_dataset.weights)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # build model
    model = models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    
    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # init gradient scaler if needed
    scaler = GradScaler() if args.use_fp16 else None

    # optionally resume from a checkpoint
    run_id = None
    args.start_epoch = 0
    
    # optionally resume from a checkpoint
    if args.resume is not None:
        print(f"=> loading checkpoint '{args.resume}'")
        
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')

        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        
        run_id = checkpoint['run_id']
        print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
        
    # log parameters for run, or resume existing run
    if run_id is None and args.rank == 0:
        # log parameters in mlflow
        mlflow.end_run()
        mlflow.set_experiment(args.experiment_name)
        mlflow.log_artifact(args.config)

        #we don't want to add everything in the config
        #to mlflow parameters, we'll just add the most
        #likely to change parameters
        mlflow.log_param('architecture', args.arch)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('batch_size', args.batch_size)
        mlflow.log_param('base_lr', args.base_lr)
        mlflow.log_param('final_lr', args.final_lr)
        mlflow.log_param('temperature', args.temperature)
        mlflow.log_param('feature_dim', args.feat_dim)
        mlflow.log_param('queue_length', args.queue_length)
        mlflow.log_param('weight_gamma', args.weight_gamma)
    else:
        # resume existing run
        mlflow.start_run(run_id=run_id)

    # build the queue, or resume it
    queue = None
    queue_path = os.path.join(args.model_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
        
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epochs):
        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue, args)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "run_id": mlflow.active_run().info.run_id,
                "norms": args.norms
            }
            if args.use_fp16:
                save_dict["scaler"] = scaler.state_dict()
                
            torch.save(
                save_dict,
                os.path.join(args.model_path, "checkpoint.pth.tar"),
            )
            
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.model_path, "checkpoint.pth.tar"),
                    os.path.join(args.model_path, f"ckp-{epoch}.pth"),
                )
                
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

def train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        if scaler is not None:
            with autocast():
                embedding, output = model(inputs)
                
                embedding = embedding.detach()
                bs = inputs[0].size(0)
                
                # ============ swav loss ... ============
                loss = 0
                for i, crop_id in enumerate(args.crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                        # time to use the queue
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                                use_the_queue = True
                                out = torch.cat((torch.mm(
                                    queue[i],
                                    model.module.prototypes.weight.t()
                                ), out))

                            # fill the queue
                            queue[i, bs:] = queue[i, :-bs].clone()
                            queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                        # get assignments
                        q = distributed_sinkhorn(out, args)[-bs:]

                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                        x = output[bs * v: bs * (v + 1)] / args.temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

                    loss += subloss / (np.sum(args.nmb_crops) - 1)

                loss /= len(args.crops_for_assign)
        else:
            embedding, output = model(inputs)
                
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.module.prototypes.weight.t()
                            ), out))

                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    q = distributed_sinkhorn(out, args)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

                loss += subloss / (np.sum(args.nmb_crops) - 1)

            loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()

            # cancel gradients for the prototypes
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # cancel gradients for the prototypes
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
                    
            optimizer.step()

        # ============ misc ... ============
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and it % args.print_freq == 0:
            progress.display(it)

    if args.rank == 0:
        # store metrics to mlflow
        mlflow.log_metric('loss', losses.avg, step=epoch)
        
    return (epoch, losses.avg), queue

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

@torch.no_grad()
def distributed_sinkhorn(out, args):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()

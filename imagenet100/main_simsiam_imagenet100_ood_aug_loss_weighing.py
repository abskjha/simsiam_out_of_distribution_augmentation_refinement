#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import sys
sys.path.append('../.') 

import random
import shutil
import time
import warnings

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

import simsiam.loader
import simsiam.builder as simsiam_builder
import torch.nn.functional as F

import time

timeStamp = time.localtime()
start_time_stamp = time.strftime("%Y%m%d_%H%M%S", timeStamp)


only_file_no_extension = os.path.basename(__file__).split('.py')[0]

from pathlib import Path
import math
import wandb
import utils

from datasets import get_dataset
import knn_monitor






wandb.init(project="simsiam_ood_aug_refinement",
          name=str(start_time_stamp)+"_"+os.path.basename(__file__),
          settings=wandb.Settings(start_method='fork'),)



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset',default='imagenet100', help='path to dataset: imagenet100')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--r-c-arch', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=int(256*1.3), type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default="env://", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--output_dir', default="/scratch/leuven/346/vsc34686/simsiam_spatial_augmentations/save/"+only_file_no_extension, type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')


##### parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
##### train_dino(args)



args = parser.parse_args()

import os 
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29511'
    

def main():
    # args = parser.parse_args()
    
    wandb.config.update(args)
    if args.seed is not None:
        utils.init_distributed_mode(args)
        utils.fix_random_seeds(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node",ngpus_per_node)
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam_builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    # initialize random resnet50
    r_model = models.__dict__[args.r_c_arch](pretrained=False)
    c_model = models.__dict__[args.r_c_arch](pretrained=False)
    
    
    # infer learning rate before changing batch size
    init_lr = args.lr * (args.batch_size * utils.get_world_size())/ 256

    if args.distributed:
        # Apply SyncBN
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            r_model.cuda(args.gpu)
            c_model.cuda(args.gpu)
            
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = args.batch_size
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            r_model = torch.nn.parallel.DistributedDataParallel(r_model, device_ids=[args.gpu])
            c_model = torch.nn.parallel.DistributedDataParallel(c_model, device_ids=[args.gpu])
        else:
            model.cuda()
            r_model.cuda()
            c_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            r_model = torch.nn.parallel.DistributedDataParallel(r_model)
            c_model = torch.nn.parallel.DistributedDataParallel(c_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        r_model = r_model.cuda(args.gpu)
        c_model = c_model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm
    
    model_without_ddp = model.module
    r_model_without_ddp = r_model.module
    c_model_without_ddp = c_model.module
    
    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    c_model_optim_params = c_model.parameters()  
    c_model_optimizer = torch.optim.SGD(c_model_optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    
    augmentation_simple = [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]

    
    
    
    dataset_kwargs = {
        'dataset':args.dataset,
        'data_dir': args.data,
        'debug_subset_size': None,
    }
    
    
    train_dataset = get_dataset( 
                    transform=simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)), 
                    train=True,
                    **dataset_kwargs)
    
    print("Length of train_dataset = ",len(train_dataset))
    
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    # print("train_dataset is loaded")
    
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    
    
    memory_dataset = get_dataset( 
                    transform= transforms.Compose(augmentation_simple), 
                    train=True,
                    **dataset_kwargs)
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
    
    val_dataset = get_dataset( 
                    transform= transforms.Compose(augmentation_simple), 
                    train=False,
                    **dataset_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
    

    for epoch in range(0, 5):
        r_c_loss_avg = ood_train(memory_loader, c_model, r_model, c_model_optimizer, epoch, args,val_loader)
    if os.path.join(args.output_dir,'r_c_models') is not None:
        Path(os.path.join(args.output_dir,'r_c_models')).mkdir(parents=True, exist_ok=True)
    save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.r_c_arch,
                'state_dict_r_model': r_model.state_dict(),
                'state_dict_c_model': c_model.state_dict(),
                'c_model_optimizer' : c_model_optimizer.state_dict(),
                'r_c_loss_avg': r_c_loss_avg,
            }, is_best=False, filename=os.path.join(args.output_dir,'r_c_models', f'r_c_model_checkpoint.pth'))
    
    # r_c_weight_file = os.path.join(args.output_dir,'r_c_models', f'r_c_model_checkpoint.pth')
    # r_c_model_checkpoint = torch.load(r_c_weight_file)
    # r_model.state_dict = r_c_model_checkpoint['state_dict_r_model'] 
    # c_model.state_dict = r_c_model_checkpoint['state_dict_c_model']
    # c_model_optimizer.state_dict = r_c_model_checkpoint['c_model_optimizer']
    # r_c_loss_avg = r_c_model_checkpoint['r_c_loss_avg']
    
    
    
    
    # print("train_loader is loaded")
    old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2 = None, None, None, None
    epoch_cv_p1, epoch_cv_z1, epoch_cv_p2, epoch_cv_z2 = None, None, None, None
    print('args.start_epoch = ',args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        
        
        # train for one epoch
        old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2 = train(train_loader, model, criterion, optimizer, epoch, args, memory_loader,val_loader, model_without_ddp, old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2,r_model,c_model, r_c_loss_avg)
        
        
        epoch_cv_p1 = update_cv_per_epoch(old_cv_p1,epoch_cv_p1)
        epoch_cv_p2 = update_cv_per_epoch(old_cv_p2,epoch_cv_p2)
        
        epoch_cv_z1 = update_cv_per_epoch(old_cv_z1,epoch_cv_z1)
        epoch_cv_z2 = update_cv_per_epoch(old_cv_z2,epoch_cv_z2)
        
        
        epoch_cv_dict = {"epoch_cv_p1": epoch_cv_p1,
                         "epoch_cv_p2": epoch_cv_p2,
                         "epoch_cv_z1": epoch_cv_z1,
                         "epoch_cv_z2": epoch_cv_z2,
                        }
        torch.save(epoch_cv_dict, os.path.join(args.output_dir, 'epoch_cv.pth'))

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
    # wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, args, memory_loader,val_loader, model_without_ddp, old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2, r_model, c_model, r_c_loss_avg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    # for r_model and c_model only eval mode
    r_model.eval()
    c_model.eval()

    end = time.time()
    
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    
    
    
    for i, (images, _) in enumerate(metric_logger.log_every(train_loader, 10, header)):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            r_out1, r_out2 = r_model(images[0]), r_model(images[1])
            c_out1, c_out2 = c_model(images[0]), c_model(images[1])
            r_c_loss1 = F.mse_loss(r_out1.detach(), c_out1.detach(),reduction='none').mean(dim=1)
            r_c_loss2 = F.mse_loss(r_out2.detach(), c_out2.detach(),reduction='none').mean(dim=1)
            r_c_bool1 = r_c_loss1 < r_c_loss_avg
            r_c_bool2 = r_c_loss2 < r_c_loss_avg
            r_c_bool12 = r_c_bool1*r_c_bool2
            
            # print(r_c_loss1.shape, r_c_loss2.shape, r_c_bool1.sum(), r_c_bool2.sum(), (r_c_bool12).sum())
        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss_naive = -((criterion(p1, z2.detach())*r_c_bool12 + criterion(p2, z1.detach())*r_c_bool12).sum()/(r_c_bool12.sum()+1e-6)) * 0.5
               
        # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        
        loss = loss_naive

        losses.update(loss.item(), images[0].size(0))
        
        
        old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2, var_p1, var_z1, var_p2, var_z2, cos_sim_old_new_cv_p1, cos_sim_old_new_cv_z1, cos_sim_old_new_cv_p2, cos_sim_old_new_cv_z2 = update_cv_per_iteration(p1,z1,p2,z2,old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2)
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(epoch=epoch)
        metric_logger.update(weight_decay=optimizer.param_groups[0]["weight_decay"])
        
        
        
        
        
        
        
        
        
        
        wandb.log({"train/loss": loss.item(),
                  "train/lr": optimizer.param_groups[0]["lr"],
                  "train/weight_decay": optimizer.param_groups[0]["weight_decay"],
                  "train/epoch": epoch + ((i + 1)/ n_steps_per_epoch),
                  "train/cos_sim_old_new_cv_z1": cos_sim_old_new_cv_z1,
                  "train/cos_sim_old_new_cv_z2": cos_sim_old_new_cv_z2,
                  "train/cos_sim_old_new_cv_p1": cos_sim_old_new_cv_p1,
                  "train/cos_sim_old_new_cv_p2": cos_sim_old_new_cv_p2,
                  "train/var_z1": var_z1,
                  "train/var_z2": var_z2,
                  "train/var_p1": var_p1,
                  "train/var_p2": var_p2,
                   "train/norm_cv_z1": torch.norm(old_cv_z1),
                   "train/norm_cv_z2": torch.norm(old_cv_z2),
                   "train/norm_cv_p1": torch.norm(old_cv_p1),
                   "train/norm_cv_p2": torch.norm(old_cv_p2),
                   
                  })
        
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # print(model_without_ddp.encoder)
    accuracy = knn_monitor.knn_monitor(net=model_without_ddp.encoder, memory_data_loader=memory_loader, test_data_loader=val_loader, epoch=epoch, k=200, hide_progress=True)
    wandb.log({"knn_evaluation/accuracy": accuracy,
               "knn_evaluation/epoch": epoch,
                })
    return old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2


def save_checkpoint(state, is_best, filename=os.path.join(args.output_dir, 'checkpoint.pth')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.output_dir, 'best_checkpoint.pth'))


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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
            
def update_cv_per_iteration(p1,z1,p2,z2,old_cv_p1=None, old_cv_z1=None, old_cv_p2=None, old_cv_z2=None):
    center_vector_z1 = z1.detach().clone().mean(0).unsqueeze(0)
    center_vector_z2 = z2.detach().clone().mean(0).unsqueeze(0)
    
    center_vector_p1 = p1.detach().clone().mean(0).unsqueeze(0)
    center_vector_p2 = p2.detach().clone().mean(0).unsqueeze(0)
    
    
    
    var_z1 = torch.var(z1.detach(), axis = 0).mean().detach()
    var_z2 = torch.var(z2.detach(), axis = 0).mean().detach()
    var_p1 = torch.var(p1.detach(), axis = 0).mean().detach()
    var_p2 = torch.var(p2.detach(), axis = 0).mean().detach()
    
    
    
    
    
    if old_cv_p1==None:
        old_cv_p1 = torch.zeros_like(center_vector_p1)
    if old_cv_z1==None:
        old_cv_z1 = torch.zeros_like(center_vector_z1)
    if old_cv_p2==None:
        old_cv_p2 = torch.zeros_like(center_vector_p2)
    if old_cv_z2==None:
        old_cv_z2 = torch.zeros_like(center_vector_z2)
    
    
    
    cos_sim_old_new_cv_z1 = F.cosine_similarity(center_vector_z1, old_cv_z1)
    cos_sim_old_new_cv_z2 = F.cosine_similarity(center_vector_z2, old_cv_z2)
    
    cos_sim_old_new_cv_p1 = F.cosine_similarity(center_vector_p1, old_cv_p1)
    cos_sim_old_new_cv_p2 = F.cosine_similarity(center_vector_p2, old_cv_p2)
    
    old_cv_p1 = center_vector_p1.clone().detach()
    old_cv_p2 = center_vector_p2.clone().detach()
    
    old_cv_z1 = center_vector_z1.clone().detach()
    old_cv_z2 = center_vector_z2.clone().detach()
        
    return old_cv_p1, old_cv_z1, old_cv_p2, old_cv_z2, var_p1, var_z1, var_p2, var_z2, cos_sim_old_new_cv_p1, cos_sim_old_new_cv_z1, cos_sim_old_new_cv_p2, cos_sim_old_new_cv_z2
    
def update_cv_per_epoch(old_cv,epoch_cv):
    if epoch_cv == None:
            epoch_cv = old_cv
    else:
        epoch_cv = torch.cat((epoch_cv, old_cv), 0)
    return epoch_cv
def ood_train(memory_loader, c_model, r_model, c_model_optimizer, epoch, args, val_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(memory_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    c_model.train()
    r_model.eval()

    end = time.time()
    
    n_steps_per_epoch = math.ceil(len(memory_loader.dataset) / args.batch_size)
    for i, (images, target) in enumerate(memory_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            r_out = r_model(images)
        c_out = c_model(images)
        loss = F.mse_loss(c_out, r_out.detach())

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        c_model_optimizer.zero_grad()
        loss.backward()
        c_model_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        wandb.log({"ood_train/loss": loss.item(),
                   "ood_train/epoch": epoch+((i+1)/n_steps_per_epoch),
                    "ood_train/lr": c_model_optimizer.param_groups[0]['lr'],
                    "ood_train/weigth_decay": c_model_optimizer.param_groups[0]['weight_decay'],
        })
            
    wandb.log({"ood_train/loss_avg": losses.avg,
               "ood_train/epoch": epoch,
                })
    if val_loader is not None:
        eval_loss_avg = ood_eval(val_loader, c_model, r_model, epoch, args)
        return eval_loss_avg
    else:
        return losses.avg

def ood_eval(val_loader, c_model, r_model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    c_model.eval()
    r_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.no_grad():
                r_out = r_model(images)
                c_out = c_model(images)
            loss = F.mse_loss(c_out.detach(), r_out.detach())

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            wandb.log({"ood_eval/loss": loss.item(),
                       "ood_eval/epoch": epoch+((i+1)/len(val_loader)),
            })
        wandb.log({"ood_eval/loss_avg": losses.avg,
                   "ood_eval/epoch": epoch,
                    })
    return losses.avg

if __name__ == '__main__':
    main()

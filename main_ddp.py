# -*- coding: utf-8 -*-
"""
@Author : Chan ZiWen
@Date : 2022/11/7 16:59
File Description:
pytorch : data distributed parallel

"""

import json
import os
import time
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.utils import NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler

from models import *
from trainer.engine import train_one_epoch, evaluate
from tools.dataset import DataGen
from tools import utils
from opts import parser
from tensorboardX import SummaryWriter


supported_model = {
    "myconvnet": ConvNet,
    "mobilevit": Mobilevit,
    "mobilevitv2_050": Mobilevitv2_050,
    "mobilevitv2_125": Mobilevitv2_125,
    "mobilevitv2_175": Mobilevitv2_175,
    "semobilevit_s": Semobilevit_s,
    "mobilenetv3_small_050": Mobilenetv3_small_050,
    "mobilenetv3_small_075": Mobilenetv3_small_075,
    "mobilenetv3_small_100": Mobilenetv3_small_100,
}


def main():
    if not os.path.exists(args.output_dir):
        print('(%s) is successfully created.'.format(args.output_dir))
        os.mkdir(args.output_dir)
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True      # 避免随机性引起的网络前馈结果的差异，此情况下也会提升计算速度

    device_type = args.device_type
    enable = False

    if device_type == 'cpu':
        device = torch.device('cpu')
    else:
        cuda_device = args.cuda_device
        device = torch.device('cuda')
        # distributed data parallel (pytorch) or accelerate (hugging face)
        enable = True if len(cuda_device) > 1 else False
    print('########################################################################')
    print('distributed mode is :', enable)
    print('########################################################################')

    if enable:
        utils.init_distributed_mode(args)

    # ============ preparing data ... ============
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    (data_loaderTrain, data_loaderEval), (lens_train, lens_val) = DataGen(
        args.dataset, enable, num_tasks=num_tasks, global_rank=global_rank)

    mixup_fn = None
    if args.dataset['mixup'] > 0. or args.dataset['cutmix'] > 0.:
        mixup_fn = Mixup(mixup_alpha=args.dataset['mixup'],
                         cutmix_alpha=args.dataset['cutmix'],
                         label_smoothing=args.dataset['smoothing'],
                         num_classes=args.model['num_classes'])

    print(f"Data Loader is ready.")

    # ============ building student and teacher networks ... ============
    mvit = supported_model[args.model_name](args.model).to(device)
    print(mvit)

    # ddp model
    if enable:
        mvit = nn.parallel.DistributedDataParallel(mvit)
        mvit_without_ddp = mvit.module
    else:
        mvit_without_ddp = mvit
        torch.cuda.empty_cache()

    # ============ preparing loss ... ============
    if args.dataset['mixup'] > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.dataset['smoothing']:
        criterion = LabelSmoothingCrossEntropy(args.dataset['smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()

    loss_scaler = NativeScaler()

    # ============ preparing optimizer ... ============
    momentum = args.momentum if 'momentum' in args.__dict__ else None
    optimizer = utils.get_optimizer(args.optimizer, mvit_without_ddp.parameters(), args.lr, momentum, args.weight_decay)

    mixed_scaler = None
    if args.mixed_precision:
        mixed_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    # niter_per_ep = lens_train // args.dataset['train_batch_size'] + \
    #                (1 if lens_train % args.dataset['train_batch_size'] != 0 else 0)
    # lr_schedule = utils.cosine_scheduler(
    #     args.lr * (args.dataset['train_batch_size'] * utils.get_world_size()) / 256.,  # linear scaling rule
    #     args.min_lr,
    #     args.epochs, niter_per_ep,
    #     warmup_epochs=args.warmup_epochs,
    # )
    lr_schedule, _ = create_scheduler(args, optimizer)
    print(f"Loss, optimizer and schedulers ready.")

    # ============  tensorboard X  ============
    summary_writer = SummaryWriter(args.logdir)

    # ============ Starting MobileViT-V1 training  ============
    start_time = time.time()
    print("Starting training!")
    progress_bar = tqdm(range(args.epochs))

    max_accuracy = 80
    # train
    for epoch in range(args.epochs):
        if args.enable:
            data_loaderTrain.sampler.set_epoch(epoch)

        # ============ training one epoch of model ... ============
        train_stats = train_one_epoch(mvit_without_ddp, criterion, data_loaderTrain, optimizer, device, epoch, loss_scaler,
                                      lr_schedule, mixup_fn=mixup_fn)

        summary_writer.add_scalar('train/loss', train_stats['loss'], epoch)
        summary_writer.add_scalar('train/lr', train_stats['lr'], epoch)

        # Necessary to pad predictions and labels for being gathered
        val_stats = evaluate(data_loaderEval, mvit, device, epoch, (1, 5))

        summary_writer.add_scalar('eval/loss', val_stats['loss'], epoch)
        summary_writer.add_scalar('eval/acc1', val_stats['acc1'], epoch)
        if val_stats.get('acc5', False):
            summary_writer.add_scalar('eval/acc5', val_stats['acc5'], epoch)

        print(f"Accuracy of the network on the {lens_val} test images: {val_stats['acc1']:.1f}%")

        progress_bar.update(1)

        # ============ writing logs ... ============
        save_dict = {
            'model': mvit.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': criterion.state_dict(),
        }
        if mixed_scaler is not None:
            save_dict['mixed_scaler'] = mixed_scaler.state_dict()

        if val_stats['acc1'] > max_accuracy:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'{args.model_name}_{epoch + 1}_{val_stats["acc1"]:.2f}.pth'))
            max_accuracy = max(max_accuracy, val_stats["acc1"])
            print(f'#############################################################  Max accuracy: {max_accuracy:.2f}%')

        # train / eval stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + '\n')

    summary_writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parser()
    print('model_name: ', args.model_name)
    print(args)
    main()


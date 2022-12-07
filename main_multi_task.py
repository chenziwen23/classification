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
from PIL import Image

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.utils import NativeScaler
from timm.scheduler import create_scheduler

from models import *
from trainer.engine import train_one_epoch_m, evaluate_m
from tools import utils
from opts import parser

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, CenterCrop


supported_model = {
    "mobilevit-xxs": ConvNet
}


class baseDataset(Dataset):
    def __init__(self, basedir, txtfile, transform=None):
        # sampler
        if isinstance(txtfile, str):
            with open(txtfile, 'r') as f:
                self.img_list = f.readlines()
        else:
            self.img_list = txtfile

        self.basedir = basedir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img2lab = self.img_list[idx].strip('\n').split(',')
        image = Image.open(os.path.join(self.basedir, img2lab[0]))

        # # 是否动漫
        # label = 1 if int(img2lab[1]) == 1 else 0
        label = int(img2lab[1])
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_data(basedir="/mnt/chenziwen/Datasets/dc/train", args_d=None):
    txtPaths = "/mnt/chenziwen/Datasets/dc/label.txt"
    f = open(txtPaths, 'r')
    img_list = f.readlines()
    lens = len(img_list)
    train_l = int(0.8 * lens)
    val_l = lens - train_l
    print(f"Train data lengths: {train_l}, Val data lengths: {val_l}")
    evalPath, trainPath = random_split(img_list, lengths=[val_l, train_l])
    f.close()

    transformTrain = Compose([
        Resize([272, 272]),
        RandomCrop([args_d['height'], args_d['width']]),
        RandomHorizontalFlip(),
        ToTensor()])
    transformTest = Compose([
        Resize([272, 272]),
        CenterCrop([args_d['height'], args_d['width']]),
        ToTensor()])
    dataTrain = baseDataset(basedir, trainPath, transform=transformTrain)
    dataEval = baseDataset(basedir, evalPath, transform=transformTest)

    # generate dataloader
    training_data_loaderTrain = DataLoader(dataset=dataTrain,
                                           num_workers=args_d['workers'],
                                           batch_size=args_d['train_batch_size'],
                                           pin_memory=args_d['pin_memory'],
                                           shuffle=True)
    # data_loaderTrain = data_prefetcher(training_data_loaderTrain)
    data_loaderTrain = training_data_loaderTrain

    training_data_loaderEval = DataLoader(dataset=dataEval,
                                          num_workers=args_d['workers'],
                                          batch_size=args_d['val_batch_size'],
                                          pin_memory=args_d['pin_memory'],
                                          shuffle=False)
    # data_loaderEval = data_prefetcher(training_data_loaderEval)
    data_loaderEval = training_data_loaderEval
    print(f"Data loaded: there are {len(dataTrain)} / {len(dataEval)} images (train / val).")
    return data_loaderTrain, data_loaderEval


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

    # ============ preparing data ... ============
    data_loaderTrain, data_loaderEval = get_data(args_d=args.dataset)
    print(f"Data Loader is ready.")

    # ============ building student and teacher networks ... ============
    mvit = supported_model[args.model_name](args.model).to(device)
    print(mvit)

    # ============ preparing loss ... ============
    criterion = nn.CrossEntropyLoss()

    # ============ preparing optimizer ... ============
    momentum = args.momentum if 'momentum' in args.__dict__ else None
    optimizer = utils.get_optimizer(args.optimizer, mvit.parameters(), args.lr, momentum, args.weight_decay)

    # ============ init schedulers ... ============
    lr_schedule, _ = create_scheduler(args, optimizer)
    print(f"Loss, optimizer and schedulers ready.")

    # ============ Starting MobileViT-V1 training  ============
    start_time = time.time()
    print("Starting training!")
    progress_bar = tqdm(range(args.epochs))

    min_loss = 0.3
    # train
    for epoch in range(args.epochs):
        # ============ training one epoch of model ... ============
        train_one_epoch_m(mvit, criterion, data_loaderTrain, optimizer, device, epoch, lr_schedule)

        # Necessary to pad predictions and labels for being gathered
        val_stats = evaluate_m(data_loaderEval, mvit, device, 1)

        progress_bar.update(1)
        # ============ writing logs ... ============
        save_dict = {
            'model': mvit.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': criterion.state_dict(),
        }
        if val_stats['loss'] < min_loss:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_{args.model_name}_{epoch + 1}.pth'))
            min_loss = min(min_loss, val_stats["loss"])
            print(f'#############################################################  '
                  f'Min loss: {min_loss:.2f}%, acc1: {val_stats["acc1"]: .2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parser()
    print('model_name: ', args.model_name)
    print(args)
    main()


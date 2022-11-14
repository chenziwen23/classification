# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/25 16:12
File Description:

"""
import os
import cv2

import albumentations
import jpeg4py as jpeg
from turbojpeg import TurboJPEG
from PIL import Image, ImageFilter
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip

interpolation_supported = {
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


class baseDataset(Dataset):
    def __init__(self, basedir, txtfile, transform=None):
        if isinstance(type(txtfile), str):
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
        image = Image.open(os.path.join(self.basedir, img2lab[0])).convert('L')
        image = image.filter(ImageFilter.FIND_EDGES).convert('RGB')
        label = int(img2lab[1])

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class jpegDataset(Dataset):
    def __init__(self, basedir, txtfile, transform=None):
        if isinstance(type(txtfile), str):
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
        image = jpeg.JPEG(os.path.join(self.basedir, img2lab[0])).decode()
        label = int(img2lab[1])

        if self.transform is not None:
            image = self.transform(**{"image": image})

        return torch.from_numpy(image['image']), torch.tensor(label, dtype=torch.long)


class turboDataset(Dataset):
    def __init__(self, basedir, txtfile, transform=None):
        if isinstance(type(txtfile), str):
            with open(txtfile, 'r') as f:
                self.img_list = f.readlines()
        else:
            self.img_list = txtfile

        self.transform = transform
        self.basedir = basedir
        self.jpeg = TurboJPEG()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img2lab = self.img_list[idx].strip('\n').split(',')
        img_b = open(os.path.join(self.basedir, img2lab[0]), 'rb').read()
        try:
            image = self.jpeg.decode(img_b)
        except OSError as O:
            print(O, ' ---- ', os.path.join(self.basedir, img2lab[0]))

        label = int(img2lab[1])

        if self.transform is not None:
            image = self.transform(**{"image": image})

        return torch.from_numpy(image['image']), torch.tensor(label, dtype=torch.long)


class data_prefetcher():
    def __init__(self, loader, fp16=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With amp, it isn't necessary to manually convert data to half.
        self.fp16 = fp16
        # if self.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_x, self.next_y = next(self.loader)
        except StopIteration:
            self.next_x, self.next_y = None, None
            return
        with torch.cuda.stream(self.stream):
            self.next_x = self.next_x.cuda(non_blocking=True)
            self.next_y = self.next_y.cuda(non_blocking=True)
            # if self.fp16:
            #     self.next_input = self.next_input.half().to(device='cuda:0', non_blocking=True)
            # else:
            # self.next_x, self.next_y = self.next_x.to(device='cuda', non_blocking=True), self.next_y.cuda(non_blocking=True)
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = (self.next_x, self.next_y)
        self.preload()
        if self.next_x is None:
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.loader)


def get_loader(args: Optional[dict], transformTrain = None, transformTest = None, dist_mode = None, **kwargs):
    num_tasks = kwargs.get('num_tasks', None)
    global_rank = kwargs.get('global_rank', None)
    basedir, txtPaths = args['root_dir'], args['txt_path']
    testPath = args.get('txt_path_test', None)
    # define dataset class
    if isinstance(txtPaths, (list, tuple)):
        trainPath, evalPath = txtPaths
    else:   # split dataset
        f = open(txtPaths, 'r')
        img_list = f.readlines()
        lens = len(img_list)
        train_l = int(0.9*lens)
        val_l = lens - train_l
        trainPath, evalPath = random_split(img_list, lengths=[train_l, val_l])
        f.close()

    dataTest = None
    dataset_supported = {
        "base": baseDataset,
        "jpeg4py": jpegDataset,
        "turbo": turboDataset,
    }
    DatasetClass = dataset_supported[args['dataset_method']]
    if args['dataset_method'] == 'base':
        transformTrain = Compose([Resize([args['height'], args['width']]),
                                  RandomHorizontalFlip(),
                                  ToTensor()])
        transformTest = Compose([Resize([args['height'], args['width']]),
                                  ToTensor()])

    dataTrain = DatasetClass(basedir, trainPath, transform=transformTrain)
    dataEval = DatasetClass(basedir, evalPath, transform=transformTest)
    if testPath:
        dataTest = DatasetClass(basedir, testPath, transform=transformTest)

    if dist_mode:
        sampler_train = DistributedSampler(dataTrain, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = DistributedSampler(dataEval, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        if testPath:
            sampler_test = DistributedSampler(dataTest, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = RandomSampler(dataTrain)
        sampler_val = SequentialSampler(dataEval)
        if testPath:
            sampler_test = SequentialSampler(dataTest)

    # generate dataloader
    training_data_loaderTrain = DataLoader(dataset=dataTrain,
                                           sampler=sampler_train,
                                           num_workers=args['workers'],
                                           batch_size=args['train_batch_size'],
                                           pin_memory=args['pin_memory'])
    # data_loaderTrain = data_prefetcher(training_data_loaderTrain)
    data_loaderTrain = training_data_loaderTrain

    training_data_loaderEval = DataLoader(dataset=dataEval,
                                          sampler=sampler_val,
                                          num_workers=args['workers'],
                                          batch_size=args['val_batch_size'],
                                          pin_memory=args['pin_memory'])
    # data_loaderEval = data_prefetcher(training_data_loaderEval)
    data_loaderEval = training_data_loaderEval
    print(f"Data loaded: there are {len(dataTrain)} / {len(dataEval)} images (train / val).")

    if testPath:
        print(f"Data loaded: there are {len(dataTest)} images (test).")
        eval_data_loaderTest = DataLoader(dataset=dataTest,
                                          sampler=sampler_test,
                                          num_workers=args['workers'],
                                          batch_size=args['eval_batch_size'],
                                          pin_memory=args['pin_memory'])
        # data_loaderTest = data_prefetcher(eval_data_loaderTest)
        data_loaderTest = eval_data_loaderTest
        return (data_loaderTrain, data_loaderEval, data_loaderTest), (len(dataTrain), len(dataEval), len(dataTest))

    # output dataloader
    return (data_loaderTrain, data_loaderEval), (len(dataTrain), len(dataEval))


def DataGen(args: Optional[dict], dist_mode = None, **kwargs):
    interpolation_method = interpolation_supported.get(args['interpolation'], 'linear')
    transform_Train = albumentations.Compose([
        albumentations.Resize(height=args['height'],
                              width=args['width'],
                              interpolation=interpolation_method,
                              always_apply=True, p=1),
        albumentations.ColorJitter(brightness=args['brightness'],
                                   contrast=args['contrast'],
                                   saturation=args['saturation'],
                                   hue=args['hue']),
        albumentations.Flip(always_apply=False),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5),
                                 max_pixel_value=255.0,
                                 always_apply=True, p=1)])

    transform_Test = albumentations.Compose([
        albumentations.Resize(height=args['height'],
                              width=args['width'],
                              interpolation=interpolation_method,
                              always_apply=True, p=1),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5),
                                 max_pixel_value=255.0,
                                 always_apply=True, p=1)])

    return get_loader(args, transform_Train, transform_Test, dist_mode, **kwargs)
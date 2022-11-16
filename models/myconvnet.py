# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/16 11:09
File Description:
fully convolutional network


"""
import os
from typing import Dict

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


def conv_bn(in_channels, out_channels, kernel_size, stride, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
        nn.BatchNorm2d(out_channels),
        )


class convNet(nn.Module):
    def __init__(self, num_classes):
        super(convNet, self).__init__()
        self.net = nn.Sequential(
            # x2
            conv_bn_relu(3, 16, 7, 1, 1),
            conv_bn_relu(16, 16, 5, 2, 1, groups=16),
            # x4
            conv_bn_relu(16, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 3, 2, 1, groups=64),
            # x8
            conv_bn_relu(64, 128, 3, 1, 1),
            conv_bn_relu(128, 128, 3, 2, 1, groups=128),
            # x16
            conv_bn_relu(64, 256, 3, 1, 1),
            conv_bn_relu(256, 256, 3, 2, 1, groups=256),
            # x32
            conv_bn_relu(256, 512, 3, 1, 1),
            conv_bn_relu(512, 512, 3, 2, 1, groups=512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # (b, 3, h, w)
        y = self.net(x)
        # (b, 512, h//32, w//32)
        y = self.pool(y)
        # (b, 512)
        y = self.classifier(y)
        return y


def ConvNet(args: Dict = None):
    model = convNet(args['num_classes'])
    if isinstance(args['finetune'], str) and os.path.exists(args['finetune']):
        finetune_pt_path = args['finetune']
        state = torch.load(finetune_pt_path, map_location='cpu')
        model.load_state_dict(state)
        print("============ loading model weight from {} ... ============".format(finetune_pt_path))
    return model

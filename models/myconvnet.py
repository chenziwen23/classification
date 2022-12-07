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
from timm.models.mobilevit import mobilevit_xxs, mobilevitv2_050, mobilevitv2_175, semobilevit_s
from timm.models.resnet import resnet50
from torch import nn


def ConvNet(args: Dict = None):
    model = mobilevit_xxs(pretrained=args['finetune'], num_classes=args['num_classes'])
    # model = resnet50(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model
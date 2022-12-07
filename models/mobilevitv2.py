# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/2 09:08
File Description:

reference paper: https://arxiv.org/abs/2206.02680

"""
from typing import Dict
from timm.models.mobilevit import mobilevitv2_050, mobilevitv2_125, mobilevitv2_175


def Mobilevitv2_050(args: Dict = None):
    model = mobilevitv2_050(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model


def Mobilevitv2_125(args: Dict = None):
    model = mobilevitv2_125(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model


def Mobilevitv2_175(args: Dict = None):
    model = mobilevitv2_175(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model
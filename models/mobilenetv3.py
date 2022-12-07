# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/12/5 15:57
File Description:

"""
from typing import Dict
from timm.models.mobilenetv3 import mobilenetv3_small_050, mobilenetv3_small_075, mobilenetv3_small_100


def Mobilenetv3_small_050(args: Dict = None):
    model = mobilenetv3_small_050(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model


def Mobilenetv3_small_075(args: Dict = None):
    model = mobilenetv3_small_075(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model


def Mobilenetv3_small_100(args: Dict = None):
    model = mobilenetv3_small_100(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model
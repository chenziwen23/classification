# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/2 09:09
File Description:

reference paper: https://arxiv.org/abs/2209.15159

"""
from typing import Dict

from timm.models.mobilevit import semobilevit_s


def Semobilevit_s(args: Dict = None):
    model = semobilevit_s(pretrained=args['finetune'], num_classes=args['num_classes'])
    return model
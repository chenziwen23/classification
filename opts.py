# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/2 11:33
File Description:
读取yaml 配置文件
"""
import os
import yaml
import argparse
from typing import Optional

Basename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

"""
read training configuration, eval parameters, and ddp , et 
"""


def parser():
    argparser = argparse.ArgumentParser("Here is configuration!")
    argparser.add_argument("--config", "-c", type=str, default="mobilevit_xxs.yaml", help="configuration path ")
    args = argparser.parse_args()
    if os.path.isfile(args.config):
        path = args.config
    else:
        path = os.path.join(Basename, args.config)
    file = open(path, 'r')
    opt = yaml.load(file, yaml.FullLoader)

    file.close()
    return argparse.Namespace(**opt)



# import torch
# import time
#
# device = "cuda:0"
#
# tensor_ = torch.randn(5*256*1024*1024, device=device)
# print(torch.cuda.memory_summary())
# time.sleep(60000)
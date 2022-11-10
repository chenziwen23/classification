# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/2 09:09
File Description:

reference paper: https://arxiv.org/abs/2209.15159

"""


from torch import nn
import torch


from mobilevit import MobileViT

args = {
    'num_classes': 2,
    'dims': [64, 80, 96],
    'transformer_blocks': [2, 4, 3],
    'channels': [16, 16, 24, 24, 48, 64, 80, 320],
    'expansion': 2,
    'finetune': "/mnt/chenziwen/cv/capreg/checkpoints/mobilevit_xxs.pt",
    'classifier_dropout': 0.1,
    'ffn_dropout': 0.0,
    'attn_dropout': 0.0,
    'dropout': 0.05,
    'number_heads': 4,
    'no_fuse_local_global_features': False,
    'conv_kernel_size': 3,
    'patch_size': 2,
    'activation': "swish",
    'normalization_name': "batch_norm_2d",
    'normalization_momentum': 0.1,
    'global_pool': "mean",
    'conv_init': "kaiming_normal",
    'linear_init': "trunc_normal",
    'linear_init_std_dev': 0.02
    }

model = MobileViT(args['dims'], args['channels'], args['num_classes'], args['transformer_blocks'], args['expansion'],
                     args['conv_kernel_size'], args['patch_size'], args['number_heads'], args)

for k, v in model.state_dict().items():
    print(k, v.shape)
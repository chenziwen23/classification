# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/24 16:08
File Description:

reference paper: https://arxiv.org/abs/2110.02178

Multi-Scale Sampler For Training Efficiency:
    Similar to CNNs, MobileViT does not require any positional embeddings and it may benefit from multi-scale inputs
during training.
    To facilitate MobileViT learn multi-scale representations without finetuning and to further improve training
efficiency (i.e., fewer optimization updates), we extend the multi-scale training method to variably-sized batch sizes

code : tools/multi_scale_sampler.py
"""
import os

import torch
from torch import nn
from einops import rearrange
from typing import List, Dict
from collections import OrderedDict

from .layers import conv_bn_relu, conv_bn_act, Transformer, Reshape, make_layer, init_layers, load_state_from_net


class MobileViTAttention(nn.Module):
    def __init__(self, in_channels: int = 3, dims: int = 512, kernel_size: int = 3, patch_size: int = 7, depth: int = 3,
                 mlp_dims: int = 1024, num_heads: int = 4, act_fn: nn.Module = None, ffn_dropout: float = 0.,
                 attn_dropout: float = 0.):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        conv_3x3_in = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     act_fn)
        conv_1x1_in = nn.Conv2d(in_channels, dims, 1, bias=False)

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        self.global_rep = Transformer(dims, depth, num_heads, 64, mlp_dims, act_fn, attn_dropout, ffn_dropout)
        self.conv_proj = conv_bn_act(dims, in_channels, 1, act_fn=act_fn)
        self.fusion = conv_bn_act(in_channels*2, in_channels, kernel_size, act_fn=act_fn)

    def forward(self, x):
        y = x.clone()       # bs, c, h, w

        ## Local Representation
        y = self.local_rep(y)       # bs, dims, h, w

        ## Global Representation
        _, _, h, w = y.shape
        y = rearrange(y, "b d (nh ph) (nw pw) -> b (nh nw) (ph pw) d", ph=self.ph, pw=self.pw,
                      nh=h//self.ph, nw=w//self.pw)          # bs, n, p, dims
        y = self.global_rep(y)   # bs, dims, h, w
        y = rearrange(y, "b (nh nw) (ph pw) d -> b d (ph nh) (pw nw)", ph=self.ph, pw=self.pw,
                      nh=h//self.ph, nw=w//self.pw)      # bs, dims, h, w

        ## Fusion
        y = self.conv_proj(y)       # bs, c, h, w
        y = torch.cat([x, y], 1)        # bs, c*2, h, w
        y = self.fusion(y)       # bs, c, h, w
        return y


class MV2Block(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int = 1, expansion: int = 4, act_fn: nn.Module = None):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expansion
        self.use_res_connect = stride == 1 and inp == oup
        if expansion != 1:
            self.conv_3x3 = nn.Sequential(OrderedDict([
                ('exp_1x1', nn.Conv2d(inp, hidden_dim, 1, bias=False)),
                ('exp_1x1_norm', nn.BatchNorm2d(hidden_dim)),
                ('exp_1x1_act', act_fn),
                ('conv_3x3', nn.Conv2d(hidden_dim, hidden_dim, 3, self.stride, padding=1, groups=hidden_dim, bias=False)),
                ('conv_3x3_norm', nn.BatchNorm2d(hidden_dim)),
                ('conv_3x3_act', act_fn),
                ('red_1x1', nn.Conv2d(hidden_dim, oup, 1, bias=False)),
                ('red_1x1_norm', nn.BatchNorm2d(oup))]))
        else:
            self.conv_3x3 = nn.Sequential(OrderedDict([
                ('conv_3x3',
                 nn.Conv2d(hidden_dim, hidden_dim, 3, self.stride, padding=1, groups=hidden_dim, bias=False)),
                ('conv_3x3_norm', nn.BatchNorm2d(hidden_dim)),
                ('conv_3x3_act', act_fn),
                ('red_1x1', nn.Conv2d(hidden_dim, oup, 1, bias=False)),
                ('red_1x1_norm', nn.BatchNorm2d(oup))]))

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv_3x3(x)
        else:
            out = self.conv_3x3(x)
        return out


class MobileViT(nn.Module):
    def __init__(self, dims: List, channels: List, num_classes: int = 1000, depths: List = None, expansion: int = 4,
                 kernel_size: int = 3, patch_size: int = 2, num_heads: int = 4, args: Dict = None):
        super().__init__()
        # make activation layer
        act_fn = make_layer(args['activation'])

        self.patch_size = patch_size

        # the first convolution layer
        self.conv_1 = conv_bn_relu(3, channels[0], kernel_size=kernel_size, stride=patch_size)

        # layer1  *2
        self.layer1 = MV2Block(channels[0], channels[1], 1, expansion, act_fn)

        # layer2  *4
        self.layer2 = nn.Sequential(MV2Block(channels[1], channels[2], 2, expansion, act_fn),
                                    MV2Block(channels[2], channels[2], 1, expansion, act_fn),
                                    MV2Block(channels[2], channels[3], 1, expansion, act_fn))

        # layer3  *8
        self.layer3 = nn.Sequential(
            MV2Block(channels[3], channels[4], 2, expansion, act_fn),
            MobileViTAttention(channels[4], dims[0], kernel_size, patch_size, depths[0], 2 * dims[0], num_heads, act_fn,
                               args['attn_dropout'], args['ffn_dropout']))

        # layer4  *16
        self.layer4 = nn.Sequential(MV2Block(channels[4], channels[5], 2, expansion, act_fn),
            MobileViTAttention(channels[5], dims[1], kernel_size, patch_size, depths[1],
                               2 * dims[1], num_heads, act_fn, args['attn_dropout'], args['ffn_dropout']))

        # layer5  *32
        self.layer5 = nn.Sequential(MV2Block(channels[5], channels[6], 2, expansion, act_fn),
            MobileViTAttention(channels[6], dims[2], kernel_size, patch_size, depths[2],
                               2 * dims[2], num_heads, act_fn, args['attn_dropout'], args['ffn_dropout']))

        self.classifier_dropout = args['classifier_dropout']
        self.classifier = nn.Sequential()
        conv_1x1_exp = conv_bn_act(channels[-2], channels[-1], 1, act_fn=act_fn)
        pool = make_layer(args['global_pool'])(1)
        self.classifier.add_module(name='conv_1x1_exp', module=conv_1x1_exp)
        self.classifier.add_module(name='pool', module=pool)
        self.classifier.add_module(name='reshape', module=Reshape(-1, channels[-1]))
        if 0. < self.classifier_dropout < 1.0:
            self.classifier.add_module(name='dropout', module=nn.Dropout(p=args['classifier_dropout'], inplace=True))
        self.classifier.add_module(name='fc', module=nn.Linear(channels[-1], num_classes))

        # initializing  the weight and bias of all layers
        if args['finetune'] is None or not os.path.exists(args['finetune']):
            init_layers(self.modules(), args)

    def forward(self, x):
        assert x.shape[-2] % self.patch_size == 0
        y = self.conv_1(x)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)

        y = self.classifier(y)
        return y


def mobilevit(args: Dict = None):
    model = MobileViT(args['dims'], args['channels'], args['num_classes'], args['transformer_blocks'], args['expansion'],
                     args['conv_kernel_size'], args['patch_size'], args['number_heads'], args)
    if isinstance(args['finetune'], str) and os.path.exists(args['finetune']):
        finetune_pt_path = args['finetune']
        model = load_state_from_net(model, finetune_pt_path)
        print("============ loading model weight from {} ... ============".format(finetune_pt_path))
    return model


def print_mem(x):
    MB = 1024. * 1024.
    print(f'        {x}         mem: ', torch.cuda.max_memory_allocated() / MB)


# if __name__ == '__main__':
# 
#     # image = '/mnt/chenziwen/cat.jpg'
#     image = '/mnt/chenziwen/Datasets/dc/test/411.jpg'
#     from PIL import Image
#     from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
# 
#     # img = Image.open(image)
#     #
#     # trans = Compose([
#     #     CenterCrop([128, 128]),
#     #     Resize([256, 256]),
#     #     ToTensor()
#     # ])
#     # img_t = trans(img)
#     # img_t = img_t.unsqueeze_(0).cuda()
#     # print(img_t.shape)
#     #
#     # from capreg.opts import parser
#     #
#     # args = parser()
#     # mvit_s = mobilevit(args.model).cuda()
#     # for m in mvit_s.modules():
#     #     print(m)
#     # mvit_s.eval()
#     # classifier
#     d = {'a': 12, 'b': 10}
#     print(d.get('c', None))


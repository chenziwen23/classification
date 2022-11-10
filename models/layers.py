# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/4 10:13
File Description:

"""
import os
import torch
from torch import nn
from einops import rearrange

from typing import List, Optional, Dict
from collections import OrderedDict


supported_conv_inits = [
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "normal",
    "trunc_normal",
]
supported_fc_inits = [
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "normal",
    "trunc_normal",
]


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ResidualConnect(nn.Module):
    def __init__(self, net: nn.Module):
        super(ResidualConnect, self).__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)


def conv_bn_relu(inp: int, oup: int, kernel_size: int = 3, stride: int = 1, bias: bool = False):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=bias)),
        ('norm', nn.BatchNorm2d(oup)),
        ('act', nn.ReLU(inplace=True))
    ]))


def conv_bn_act(inp: int, oup: int, kernel_size: int = 3, stride: int = 1, act_fn: nn.Module = None, bias: bool = False):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=bias)),
        ('norm', nn.BatchNorm2d(oup)),
        ('act', act_fn)
    ]))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.lnorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.lnorm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout_ratio, act_fn):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            act_fn,
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ffn(x)


class MHSA(nn.Module):
    def __init__(self, dim, heads, head_dim: int = None, dropout_ratio: float = 0.):
        super(MHSA, self).__init__()
        if head_dim is None:
            head_dim = dim // heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout_ratio, inplace=True)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x shape is [b, n, p, d]
        qkv = self.to_qkv(x)
        q, k, v = map(lambda t: rearrange(t, 'b n p (h d) -> b (n h) p d', h=self.heads), qkv.chunk(3, dim=-1))
        q = torch.einsum('bNpd, bNdP -> bNpP', q, k.transpose(-1, -2)) * self.scale
        q = self.attend(q)
        q = torch.einsum('bNpP, bNPd -> bNpd', q, v)
        q = rearrange(q, "b (n h) p d -> b n p (h d)", h=self.heads)
        return self.to_out(q)


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, head_dims, mlp_dim, act_fn = None, drop_r_att = 0., drop_r_ffn = 0.):
        super().__init__()
        self.layers = nn.Sequential()
        self.depth = depth

        for i in range(depth):
            self.layers.add_module(name=f'pre_norm_mha_{i}', module=ResidualConnect(nn.Sequential(
                nn.LayerNorm(dim),
                MHSA(dim, num_heads, head_dims, drop_r_att)))
            )
            self.layers.add_module(name=f'pre_norm_ffn_{i}', module=ResidualConnect(nn.Sequential(
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, drop_r_ffn, act_fn)))
            )
        self.layers.add_module(name='norm', module=nn.LayerNorm(dim))

    def forward(self, x):
        return self.layers(x)


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, num_heads, head_dims, mlp_dim, act_fn = None, drop_r_att = 0., drop_r_ffn = 0.):
#         super().__init__()
#         layer = []
#         self.depth = depth
#         for _ in range(depth-1):
#             layer.append(nn.ModuleList(
#                         [PreNorm(dim, Attention(dim, num_heads, head_dims, drop_r_att)),
#                          PreNorm(dim, FeedForward(dim, mlp_dim, drop_r_ffn, act_fn))]))
#         layer.append(nn.ModuleList(
#             [PreNorm(dim, Attention(dim, num_heads, head_dims, drop_r_att)),
#              PreNorm(dim, FeedForward(dim, mlp_dim, drop_r_ffn, act_fn)),
#              nn.LayerNorm(dim)]))
#         self.layers = nn.ModuleList(layer)
#
#     def forward(self, x):
#         out = x
#         for i in range(self.depth-1):
#             out = out + self.layers[i][0](out)
#             out = out + self.layers[i][1](out)
#         out = out + self.layers[-1][0](out)
#         out = out + self.layers[-1][1](out)
#         out = self.layers[-1][2](out)
#         return out


def make_layer(layer_name: str = None):
    layer_name = layer_name.lower()
    # activation  : return class instance
    if layer_name == "relu":
        return nn.ReLU(inplace=True)
    elif layer_name == "prelu":
        return nn.PReLU()
    elif layer_name == "gelu":
        return nn.GELU()
    elif layer_name == "swish":
        return nn.SiLU(inplace=True)
    elif layer_name == "mish":
        return nn.Mish(inplace=True)

    # pool layer  : return class
    elif layer_name == "mean":
        return nn.AdaptiveAvgPool2d
    elif layer_name == "max":
        return nn.AdaptiveMaxPool2d

    else:
        raise ValueError("Not Found this layer (%s).".format(layer_name))


def _init_nn_layers(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = None) -> None:
    """
    Helper function to initialize neural network module
    """
    init_method = init_method.lower()
    if init_method == "kaiming_normal":
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "kaiming_uniform":
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_normal":
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_uniform":
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "trunc_normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = "Supported initialization methods are:"
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += "\n \t {}) {}".format(i, l)
        raise ValueError("{} \n Got: {}".format(supported_conv_message, init_method))


def init_layers(module: nn.Module, args: Optional[Dict]):
    # conv_init, linear_init, linear_init_std_dev = , args['linear_init'], args['linear_init_std_dev']
    for m in module:
        if isinstance(m, nn.Sequential):
            for sm in m:
                if isinstance(sm, (nn.Conv2d, nn.Conv3d)):
                    if 'conv_init_std_dev' in args:
                        conv_init_std_dev = args['conv_init_std_dev']
                    else:
                        conv_init_std_dev = None
                    _init_nn_layers(sm, args['conv_init'], conv_init_std_dev)
                elif isinstance(sm, nn.Linear):
                    _init_nn_layers(sm, args['linear_init'], args['linear_init_std_dev'])
                elif isinstance(sm, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(sm.weight)
                    if hasattr(sm, 'bias') and sm.bias is not None:
                        nn.init.zeros_(sm.bias)
        else:
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                if 'conv_init_std_dev' in args:
                    conv_init_std_dev = args['conv_init_std_dev']
                else:
                    conv_init_std_dev = None
                _init_nn_layers(m, args['conv_init'], conv_init_std_dev)
            elif isinstance(m, nn.Linear):
                _init_nn_layers(m, args['linear_init'], args['linear_init_std_dev'])
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)


def load_state_from_net(model: torch.nn.Module, path: str = None) -> nn.Module:
    # load weight
    if os.path.isfile(path):
        net_state_dict = torch.load(path, map_location='cpu')
    else:
        raise ValueError("Can't find the file path (%s)".format(path))

    model_s = model.state_dict()
    msw_fc_key = list(model_s)[-2]
    msb_fc_key = list(model_s)[-1]
    nsw_fc_key = list(net_state_dict)[-2]
    nsb_fc_key = list(net_state_dict)[-1]
    if model_s[msb_fc_key].shape == net_state_dict[nsb_fc_key].shape:
        # if num_classification is equal
        model.load_state_dict(net_state_dict)
    else:
        # if num_classification is not equal
        model_state_dict = {}

        for ms, ns in zip(model_s, net_state_dict):
            if ns in [nsw_fc_key, nsb_fc_key]:
                continue
            assert model_s[ms].shape == net_state_dict[ns].shape, \
                ValueError(f"({ms} : {list(model_s[ms].shape)}) isn't matched to net_state({ns} : "
                           f"{list(net_state_dict[ns].shape)})")
            model_state_dict[ms] = net_state_dict[ns]
        model.state_dict().update(model_state_dict)
    return model


def count_paratermeters(model: nn.Module = None):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
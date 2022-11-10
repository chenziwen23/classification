# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/29 14:19
File Description:

"""
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# 实现思路1： 低效
class PatchEmbedding1(nn.Module):
    def __init__(self, in_channel: int = 3, patch_size: int = 16, embed_size: int = 768 ):
        super(PatchEmbedding1, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # break-down the image in h x w patches which size is s1 x s2 and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c) ', s1=patch_size, s2=patch_size, c=in_channel),
            nn.Linear(patch_size*patch_size*in_channel, embed_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


# 实现思路2： 高效
class PatchEmbedding(nn.Module):
    def __init__(self, in_channel: int = 3, patch_size: int = 16, embed_size: int = 768, img_size: int = 224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channel, embed_size, patch_size, patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        # Position Embedding: (img_size // patch_size) ** 2 + 1,    img_size // patch_size = h
        self.positions = nn.Parameter(torch.rand((img_size//patch_size)**2 + 1, embed_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)          #   (b, n, e) ,   n = h*w
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)      #   (b, 1, e)

        # prepend the cls token to the input
        x = torch.cat([cls_token, x], dim=1)
        # add position embedding:    (n + 1, e)
        x += self.positions
        # return : (b, n+1, e)
        return x


class MultiHeadAttention1(nn.Module):
    def __init__(self, embed_size: int = 768, num_heads: int = 8, dropout_ratio: float = 0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout_ratio)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)
        self.scaling = (self.embed_size // num_heads) ** 0.5

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b n h d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b n h d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b n h d", h=self.num_heads)

        # sum up over the last axis, multiply
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1) / self.scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# Note we can use a single matrix to compute in one shot queries, keys and values.
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int = 768, num_heads: int = 8, dropout_ratio: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.att_drop = nn.Dropout(dropout_ratio)
        self.to_qkv = nn.Linear(embed_size, embed_size*3)
        self.projection = nn.Linear(embed_size, embed_size)
        self.scaling = embed_size ** 0.5

    def forward(self, x: Tensor, mask: Tensor = None):
        qkv = rearrange(self.to_qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        # batch, num_heads, query_len, key_len     ,   query_len: patch nums
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1) / self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum("bhqk, bhkd -> bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        return self.fn(x, **kwargs) + res


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4, drop_r: float = 0.):
        super().__init__(
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_r),
            nn.Linear(embed_size * expansion, embed_size)
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 embed_size: int = 768,
                 drop_r: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_r: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embed_size),
                MultiHeadAttention(embed_size, **kwargs),
                nn.Dropout(drop_r)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embed_size),
                FeedForwardBlock(embed_size, forward_expansion, forward_drop_r),
                nn.Dropout(drop_r)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, embed_size=768, n_classes: int = 1000):
        super().__init__(
            Reduce("b n e -> b e", reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, n_classes)
        )


class VIT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embed_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, embed_size, img_size),
            TransformerEncoder(depth, embed_size=embed_size, **kwargs),
            ClassificationHead(embed_size, n_classes)
        )


# if __name__ == '__main__':
#     img = Image.open('/mnt/chenziwen/cat.jpg')
#     # img = Image.open('/Users/chenziwen/Pictures/cat.jpg')
#     # img.show()
#     # fig = plt.figure()
#     # plt.imshow(img)
#     # plt.show()
#
#     transform = Compose([Resize((224, 224)),
#                          ToTensor()])
#     x = transform(img)
#     x = x.unsqueeze(0)
#     patches_embedded = PatchEmbedding()(x)
#     y = TransformerEncoderBlock()(patches_embedded).shape
#     print(y)
#     summary(VIT(), (3, 224, 224), device='cpu')
# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/8 14:05
File Description:

"""
import math
import numpy as np
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, RandomSampler


# MobileVIT Multi-scale sampler
class MultiScaleSamplerDDP(Sampler):
    def __init__(self, base_im_w: int, base_im_h: int, base_batch_size: int, n_data_samples: int,
                 min_scale_mult: float=0.5, max_scale_mult: float=1.5, n_scales: int=5, is_training: bool=False) -> None:
        # min, and max, spatial dimensions
        min_im_w, max_im_w = int(base_im_w * min_scale_mult), int(base_im_w * max_scale_mult)
        min_im_h, max_im_h = int(base_im_h * min_scale_mult),  int(base_im_h * max_scale_mult)

        # Get the GPU and mode related information
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()

        # adjust the total samples to avoid batch dropping
        num_samples_per_replicas = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replicas * num_replicas
        img_indices = [idx for idx in range(n_data_samples)]
        img_indices *= img_indices[:(total_size - n_data_samples)]
        assert len(img_indices) == total_size
        self.shuffle = False

        if is_training:
            # compute the spatial dimension and corresponding batch size
            width_dims = list(np.linspace(min_im_w, max_im_w, n_scales))
            height_dims = list(np.linspace(min_im_h, max_im_h, n_scales))
            # ImageNet models down-sample images by a factor of 32
            # ensure that width and height dimensions are multiple of 32
            width_dims = [(w // 32) * 32 for w in width_dims]
            height_dims = [(h // 32) * 32 for h in height_dims]

            img_batch_pairs = list()



    def __iter__(self, ) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]


"""

if self. shuffle:
random. seed(self.epoch)
random. shuffle ‹ self.ime.indices\
random. shuffle (self.img-batch-pairs)
indices.rank. - self.img.indices|self.ram
self.img.indices):self.sum.replicas|
else
indices.rank.i = self.img.indices|self.rank
:lem(self.img.indices): self.mum_replicas]
start.indes = 0
while start_index @ self..samples.per_replica
curr_h
curr_bsz random. choice ( self. img-batch-pairs)
end_index mind start_index + curr_bsz, self
-m-samples-per.replica)
batch_ids indices.rank_i|start.index: end_index]
n_batch.samples = len(batch.ids)
if m.batch.samples e cure.bsz
batch_ids += indices_rank.il: curr.bsz n_batch.samples)]
start.index
curt_bsz
if lendhatch ids\ S
0:
batch - (curr.h, curr.w, b.id) for b.id
yield hatch
batch.ids I
def set.epoch(self, epoch: int) -> None:
self. epoch a epoch
"""
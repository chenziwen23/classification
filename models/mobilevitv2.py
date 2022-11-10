# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/2 09:08
File Description:

reference paper: https://arxiv.org/abs/2206.02680

"""
import time

import torch
import torch.nn as nn


class MobileViTV2(nn.Module):
    def __init__(self):
        super(MobileViTV2, self).__init__()


    def forward(self, x):
        return


def print_mem(x):
    MB = 1024. * 1024.
    print(f'        {x}         mem: ', torch.cuda.max_memory_allocated() / MB)


if __name__ == '__main__':
    # pt = torch.load("/mnt/chenziwen/cv/capreg/checkpoints/mobilevit_xxs.pt")
    # for k, v in pt.items():
    #     print(k, v.shape)
    # print('sssss')
    # time.sleep(20)
    # from layers import Attention
    #
    # att1 = Attention(64, 4, 64*2, 0.).cuda()
    # att2 = Attention(64, 4, 64*2, 0.).cuda()
    #
    # x = torch.randn(200, 4, 1024, 64).cuda()
    # y = att1(x)
    # print_mem('out_ -2')        # 0.85205078125
    # z = att2(y)
    # print_mem('out_ -2')  # 1.2734375
    time.sleep(10)
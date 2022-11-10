# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/28 16:33
File Description:

"""
import torch.nn as nn

loss_name = {
    "CE": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
}


"""Binary CE for classification tasks"""
def criterion(name):
    return loss_name[name]


# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/3 16:57
File Description:

"""
import glob
import os
import torch
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.mobilevit import MobileViT


# show unzip dir
train_dir = '/mnt/chenziwen/Datasets/dc/train'
test_dir = '/mnt/chenziwen/Datasets/dc/test'

print('len:', len(os.listdir(train_dir)), len(os.listdir(test_dir)))

batch_size = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(1234)

if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

lr = 0.001

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print('show data:', len(train_list), train_list[:3])
print('show data:', len(test_list), test_list[:3])

train_list, val_list = train_test_split(train_list, test_size=0.2)
print(len(train_list), train_list[:3])
print(len(val_list), val_list[:3])

train_transforms = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.RandomCrop((256, 256)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])


class dataset(Dataset):
    def __init__(self, file_list, now_transform):
        self.file_list = file_list  # list of path
        self.transform = now_transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        #         print(img.size)
        img_transformed = self.transform(img)

        # test 没有标签?
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        else:
            assert False

        return img_transformed, label


train_data = dataset(train_list, train_transforms)
val_data = dataset(val_list, test_transforms)
# test_data = dataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))
print(len(val_data), len(val_loader))

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
model = model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=lr)
loss_f = nn.CrossEntropyLoss()

epochs = 20

print('start epoch iter, please wait...')
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    model.train()
    best_acc = 0.65
    s = time.time()
    print('train')
    for data, label in train_loader:
        print('load data')
        data = data.to(device)
        label = label.to(device)
        print('data, label')
        output = model(data)
        print('output')
        loss = loss_f(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        print('-', end=' ')
    print('\nEpoch : {}, train accuracy : {}, train loss : {}, time: {}'.format(
        epoch + 1, epoch_accuracy, epoch_loss, time.time() - s))

    s = time.time()
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = loss_f(val_output, label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)
        if epoch_val_accuracy > best_acc:
            best_acc = epoch_val_accuracy
            torch.save(model.state_dict(), '/mnt/chenziwen/cv/capreg/mobilevit_xxs_dc.pt')
        print('      :   , val_accuracy : {}, val_loss : {}, time: {}'.format(
            epoch + 1, epoch_val_accuracy, epoch_val_loss, time.time()-s))
# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/6 15:40
File Description:

"""
import os
import random
import shutil
import sys
import time

import cv2
from io import BytesIO

import numpy as np
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True


def img_trans(src, dst):
    if os.path.isfile(src):
        dis_img = os.path.join(dst, os.path.basename(src).replace('png', 'jpg'))
        try:
            img = Image.open(src).convert("RGB")
            img_new = img.resize([960, 540], Image.BILINEAR)
        except OSError:
            with open(src, 'rb') as f:
                f = f.read()
            f = f + B'\xff' + B'\xd9'
            img = Image.open(BytesIO(f)).convert("RGB")
            img_new = img.resize([960, 540], Image.BILINEAR)
        img_new.save(dis_img, quality=100)
    else:
        if not os.path.exists(dst):
            os.mkdir(dst)
        imgs_src = os.listdir(src)
        for img_src in imgs_src:
            dis_img = os.path.join(dst, img_src.replace('png', 'jpg'))
            if 'png' not in img_src or os.path.exists(dis_img):
                continue

            try:
                img = Image.open(os.path.join(src, img_src)).convert("RGB")
                img_new = img.resize([960, 540], Image.BILINEAR)
            except OSError:
                with open(os.path.join(src, img_src), 'rb') as f:
                    f = f.read()
                f = f + B'\xff' + B'\xd9'
                img = Image.open(BytesIO(f)).convert("RGB")
                img_new = img.resize([960, 540], Image.BILINEAR)
            # print(img_new.size)
            img_new.save(dis_img, quality=100)


def gen_txt_dc(root='/mnt/chenziwen/Datasets/dc/train', path_label='/mnt/chenziwen/Datasets/dc/label.txt'):

    img_name_list = os.listdir(root)
    txt_content = []
    for img_n in img_name_list:
        # if 'dog' in img_n:
        #     txt_content.append(f"{img_n},{1}\n")
        # elif 'cat' in img_n:
        #     txt_content.append(f"{img_n},{0}\n")
        # else:
        #     raise ValueError(f"The image({img_n}) couldn't recognized!")
        if img_n.split('.')[-1] in ['png', 'jpg']:
            txt_content.append(f"{img_n},{0}\n")

    with open(path_label, 'w') as f:
        f.writelines(txt_content)
    print(f'path_label is written in {path_label}')


def tool(file_p, name, x, y):
    img_o = Image.open(file_p)
    img = img_o.filter(ImageFilter.FIND_EDGES)
    std0 = np.std(img)
    mean0 = np.mean(img)
    x.append([std0, mean0])
    y.append(0 if int(name.split('.')[0]) >= 10000 else 1)


def prepare():
    root = '/Users/chenziwen/Downloads/CaptureIMGS/images_mini'
    img_list = os.listdir(root)
    x_feats = []
    y = []
    x_feats_t = []
    y_t = []
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(16)
    random.seed(0)
    for name in img_list:
        if 'jpg' not in name:
            continue
        file_p = os.path.join(root, name)
        if random.random() < 0.8:
            # features.append(executor.submit(tool, file_p, name, x_feats, y))
            executor.submit(tool, file_p, name, x_feats, y)
        else:
            # features.append(executor.submit(tool, file_p, name, x_feats_t, y_t))
            executor.submit(tool, file_p, name, x_feats_t, y_t)
    executor.shutdown(wait=True)

    return np.array(x_feats), np.array(y), np.array(x_feats_t), np.array(y_t)


def main():
    img_trans(src, dst)
    # gen_txt_dc(src, '/Users/chenziwen/Downloads/CaptureIMGS/watching.txt')


if __name__ == '__main__':
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/images"
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/background"
    src = "/Volumes/KINGSTON/test"
    dst = "/Users/chenziwen/Downloads/CaptureIMGS/images_mini"
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    # dst = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    main()


    file_p = "/Users/chenziwen/Downloads/CaptureIMGS/images_mini/00000831.jpg"
    # file_p = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    # img_o = cv2.imread(file_p)
    # new_h, new_w = img_o.shape[0] // 3, img_o.shape[1] // 3
    # img_o = cv2.resize(img_o, dsize=(new_w, new_h))
    # cv2.imshow(' ss', img_o)
    # cv2.waitKey(0)

    # img = Image.open(file_p)
    # img1 = img.filter(ImageFilter.FIND_EDGES)
    # img3 = img.convert('L').filter(ImageFilter.FIND_EDGES)
    #
    # img1.show('s')
    # img3.show('3333')


# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/25 16:13
File Description:

"""
import os
import time
import glob
import random

import cv2
import numpy as np
import torch.onnx
from PIL import Image


def dir2txt(dir: str = None, class_name2idPath: str = None, txt_dir: str = './'):
    """
    img label
    :return: None
    """
    class_names = []
    class_names2id = {}
    k = 0
    for i in os.listdir(dir):
        if '.' not in i:
            class_names.append(i)
            class_names2id[i] = k
            k += 1

    if class_name2idPath is not None:
        with open(class_name2idPath, 'w') as f:
            for k, v in class_names2id.items():
                f.write(str(k) + ' ' + str(v) + '\n')

    trainp = os.path.join(txt_dir, 'train.csv')
    evalp = os.path.join(txt_dir, 'eval.csv')
    testp = os.path.join(txt_dir, 'test.csv')

    img_paths = []
    for k in class_names:
        x = glob.glob(os.path.join(dir, k, '*', '*.jpg'))
        x = [[i.lstrip(dir+'/'), class_names2id[k]] for i in x]
        img_paths.extend(x)

    random.shuffle(img_paths)
    a, b = int(len(img_paths) * 0.7), int(len(img_paths) * 0.8)

    with open(trainp, 'w') as f:
        for k in img_paths[:a]:
            f.write(str(k[0]) + ' ' + str(k[1]) + '\n')
    with open(evalp, 'w') as f:
        for k in img_paths[a:b]:
            f.write(str(k[0]) + ' ' + str(k[1]) + '\n')
    with open(testp, 'w') as f:
        for k in img_paths[b:]:
            f.write(str(k[0]) + ' ' + str(k[1]) + '\n')


def dir2txt_m(dir: str = None, txt_dir: str = './'):
    """
    img label
    :return: None
    """
    # class_names2id = {}
    # if class_name2idPath is not None:
    #     with open(class_name2idPath, 'r') as f:
    #         for k, v in class_names2id.items():
    #             f.write(str(v) + ',' + str(k) + '\n')
    # else:
    #     class_names2id = {0: "background", 1: "cartoon", 2: "modern", 3: "ancient"}

    trainp = os.path.join(txt_dir, 'train.txt')
    evalp = os.path.join(txt_dir, 'eval.txt')
    # testp = os.path.join(txt_dir, 'test.txt')

    # img_paths = glob.glob(os.path.join(dir, '*.jpg'))
    img_paths = os.listdir(dir)

    random.shuffle(img_paths)
    a = int(len(img_paths) * 0.8)

    with open(trainp, 'w') as f:
        for k in img_paths[:a]:
            if '_' not in k:
                continue
            l = int(k.split('_')[0])
            f.write(k + ',' + str(l) + '\n')
    with open(evalp, 'w') as f:
        for k in img_paths[a:]:
            if '_' not in k:
                continue
            l = int(k.split('_')[0])
            f.write(k + ',' + str(l) + '\n')
    # with open(testp, 'w') as f:
    #     for k in img_paths[b:]:
    #         f.write(str(k[0]) + ',' + str(k[1]) + '\n')


"""
name_id_path = '/Users/chenziwen/Downloads/CaptureIMGS/tv_logo_name_id.txt'
id_name_path = '/Users/chenziwen/Downloads/CaptureIMGS/tv_logo_id_name.txt'
i = 0
name_id = {}
id_name = {}
for p in os.listdir(tv_logo):
    if 'png' not in p:
        continue
    name_id[p.split('.')[0]] = i
    id_name[i] = p.split('.')[0]

    i += 1

with open(name_id_path, 'w') as f:
    for k, v in id_name.items():
        f.write(str(k) + ',' + str(v) + '\n')
with open(id_name_path, 'w') as f:
    for k, v in name_id.items():
        f.write(str(k) + ',' + str(v) + '\n')
print(id_name)
print(name_id)
"""


#################   ################
def gen_composed_logo_image(logo_dir='/mnt/chenziwen/Datasets/tvlogo/tvlogo',
                            name_id_path='/mnt/chenziwen/Datasets/tvlogo/tv_logo_name_id.txt',
                            save_dir='/mnt/chenziwen/Datasets/tvlogo/images',
                            base_images_dir='/mnt/chenziwen/Datasets/movies/movie_imgs_640'):
    max_l_logo = 192
    max_l_bg = 256
    random.seed(1234)
    name_id = {}
    with open(name_id_path, 'r') as f:
        res = f.readlines()
        res = [tuple(x.strip('\n').split(',')) for x in res]
        # f.write(str(k) + ',' + str(v) + '\n')
    name_id = dict(res)

    base_img_names = os.listdir(base_images_dir)
    for imgp in os.listdir(logo_dir):
        if 'png' not in imgp:
            continue
        id = name_id[imgp.split('.')[0]]
        imgap = os.path.join(logo_dir, imgp)
        img = Image.open(imgap).convert('RGBA')
        img = resize(img, max_l_logo)
        _, _, _, a = img.split()

        base_images_sampler = random.sample(base_img_names, 520)
        i = 0
        for base_img_name in base_images_sampler:
            img_base = Image.open(os.path.join(base_images_dir, base_img_name))
            box = get_box(img_base.size, max_l_bg)
            img_base = img_base.crop(box=box)
            w, h = img.size
            box = (int((max_l_bg - w)/2.), int((max_l_bg - h)/2.))
            img_base.paste(img, box=box, mask=a)
            img_base.save(save_dir + f'/{int(id):>03}_{i:>04}.png')
            i += 1
    return


def resize(img: Image, max_l=192, min_l=128):
    w, h = img.size
    num_random = random.random()
    if h > max_l and w > max_l:
        l = int((max_l - min_l) * num_random) + min_l
        ratio = min(l/w, l/h)
        nw, nh = int(ratio * w), int(ratio * h)
        return img.resize((nw, nh))
    else:
        if num_random > 0.6:
            ratio = min(min_l / w, min_l / h)
            nw, nh = int(ratio * w), int(ratio * h)
            return img.resize((nw, nh))
    return img


def get_box(hw, l=256):
    w, h = hw
    assert h > l and w > l, ValueError(f'the selected length({l}) > image size.')
    nh = random.randint(0, h-l)
    nw = random.randint(0, w-l)
    return (nw, nh, nw+l, nh+l)


if __name__ == '__main__':
    # dir2txt('/Users/chenziwen/Downloads/MFRD', txt_dir='./')

    # train = '/mnt/chenziwen/proserve/train.csv'
    # test = '/mnt/chenziwen/proserve/test.csv'
    # file2 = '/Users/chenziwen/Downloads/id2mac2label.csv'
    # dir2txt_m('/mnt/chenziwen/Datasets/movies/movie_imgs_640', txt_dir='/mnt/chenziwen/Datasets/movies')

    # tv_logo = '/Users/chenziwen/Downloads/CaptureIMGS/tvlogo'
    # max_l_bg = 256
    # imgap = os.path.join(tv_logo, '纯享4K.png')
    # img = Image.open(imgap).convert('RGBA')
    # img = resize(img, 192)
    # r, g, b, a = img.split()
    #
    # base_image = '/Users/chenziwen/Downloads/CaptureIMGS/images_mini/00000407.jpg'
    # img_base = Image.open(base_image)
    # box = get_box(img_base.size, 256)
    # img_base = img_base.crop(box=box)
    #
    # w, h = img.size
    # box = (int((max_l_bg - w) / 2.), int((max_l_bg - h) / 2.))
    # img_base.paste(img, box=box, mask=a)
    # img_base.show('s6.png')

    # gen_composed_logo_image()
    # dir2txt_m('/mnt/chenziwen/Datasets/tvlogo/images', '/mnt/chenziwen/Datasets/tvlogo')

    def dir2tx2t(dir: str = None, txt_dir: str = './'):
        trainp = os.path.join(txt_dir, 'train.txt')
        evalp = os.path.join(txt_dir, 'eval.txt')
        img_paths = os.listdir(dir)

        random.shuffle(img_paths)
        a = int(len(img_paths) * 0.8)

        with open(trainp, 'w') as f:
            for k in img_paths[:a]:
                if '_' not in k:
                    continue
                l = int(k.split('_')[0])
                f.write(k + ',' + str(l) + '\n')
        with open(evalp, 'w') as f:
            for k in img_paths[a:]:
                if '_' not in k:
                    continue
                l = int(k.split('_')[0])
                f.write(k + ',' + str(l) + '\n')
    dir2tx2t('/mnt/chenziwen/Datasets/dc/train', '/mnt/chenziwen/Datasets/dc')



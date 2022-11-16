# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/25 21:47
File Description:

"""

import os
import glob
import random


def rename(p):
    # rename random string
    imgs_p = os.listdir(p)
    for i in range(len(imgs_p)):
        imgp = os.path.join(p, imgs_p[i])
        n = imgs_p[i].split('.')[0]
        if n == '':
            continue
        imgp_new = os.path.join(p, f"{int(n):0>8d}.png")
        os.rename(imgp, imgp_new)
        print(f'rename successfully', imgp_new)

    # # rename special string
    # imgs_p = os.listdir(p)
    # for i in range(len(imgs_p)):
    #     imgp = os.path.join(p, imgs_p[i])
    #     imgp_new = os.path.join(p, f"{i}.png")
    #     os.rename(imgp, imgp_new)
    #     print(f'rename successfully', imgp_new)


# 不同文件夹，且需要同步其他文件夹内同文件名字
def syncfile(dl, watches, watchnoes):
    counts = 0
    for i in dl:
        print(i)
        imgsN = os.listdir(i)
        for j in range(len(imgsN)):
            if "png" not in imgsN[j]:
                counts -= 1
                continue
            imgsP = os.path.join(i, imgsN[j])
            imgsW = os.path.join(watches, imgsN[j])
            id = int(imgsN[j].split('.')[0])
            newN = f"{id:0>8d}.png"
            imgsPN = os.path.join(i, newN)
            imgsWN = os.path.join(watches, newN)
            if not os.path.exists(imgsW):
                print('failed')
                print(f"{imgsP} and {imgsW}")
                continue
            # if imgsP != imgsPN:
            #     os.rename(imgsP, imgsPN)
            # if imgsW != imgsWN:
            #     os.rename(imgsW, imgsWN)
        counts += len(imgsN)
    print('over ')
    imgsN = os.listdir(watchnoes)
    for j in range(len(imgsN)):
        if "png" not in imgsN[j]:
            continue
        imgsP = os.path.join(watchnoes, imgsN[j])
        id = int(imgsN[j].split('.')[0])
        newN = f"{id:0>8d}.png"
        imgsPN = os.path.join(watchnoes, newN)
        # if imgsP != imgsPN:
        #     os.rename(imgsP, imgsPN)
    print('over ')


# 不同文件夹，且需要同步其他文件夹内同文件名字  part 2 ： 按顺序命名
def syncfile2(dl, watches, watchnoes):
    counts = 0
    ins = []
    for i in dl:
        print(i)
        count = 0
        imgsN = os.listdir(i)
        for j in range(len(imgsN)):
            if "png" not in imgsN[j]:
                continue
            imgsP = os.path.join(i, imgsN[j])
            imgsW = os.path.join(watches, imgsN[j])
            newN = f"{j+counts}.png"

            imgsPN = os.path.join(i, newN)
            imgsWN = os.path.join(watches, newN)
            if not os.path.exists(imgsW) or imgsW in ins:
                print('failed')
                print(f"{imgsP} and {imgsW}")
                continue
            ins.append(imgsW)
            count += 1
            print(f"{imgsPN} -> {imgsWN}")
            if imgsP != imgsPN:
                os.rename(imgsP, imgsPN)
            if imgsW != imgsWN:
                os.rename(imgsW, imgsWN)

        counts += count
        print(counts, len(imgsN))
    print(f'over , {counts}')
    imgsN = os.listdir(watchnoes)
    for j in range(len(imgsN)):
        if "png" not in imgsN[j]:
            continue
        imgsP = os.path.join(watchnoes, imgsN[j])
        newN = f"{j+counts}.png"
        imgsPN = os.path.join(watchnoes, newN)
        print(f"{imgsP} -> {imgsPN}")
        if imgsP != imgsPN:
            os.rename(imgsP, imgsPN)
    print('over ')


def gen_txt(dl, watch, name2label, name2label_w, root="/Users/chenziwen/PycharmProjects/videoDet/capreg/"):
    label_watchs = []
    label_cls = []
    for pathd in dl:
        name = os.path.split(pathd)[-1]
        labid = name2label[name]

        # imgs = glob.glob(pathd + "/*.png")
        imgs = os.listdir(pathd)
        for i in imgs:
            if 'png' not in i:
                continue
            label_cls.append([i, str(labid)])

    wl = glob.glob(watch + "/*")
    for pathd in wl:
        name = os.path.split(pathd)[-1]
        labid = name2label_w[name]

        imgs = os.listdir(pathd)
        for i in imgs:
            if 'png' not in i:
                continue
            label_watchs.append([i, str(labid)])
        # os.path.join(pathd)
    random.shuffle(label_cls)
    random.shuffle(label_watchs)
    print(label_cls)
    print(label_watchs)

    # save
    with open(root+'label_watch.txt', 'w') as lw:
        for line in label_watchs:
            lw.writelines(','.join(line) + '\n')

    with open(root+'label_class.txt', 'w') as lc:
        for line in label_cls:
            lc.writelines(','.join(line) + '\n')


if __name__ == '__main__':
    label2name = {0: "background", 1: "cartoon", 2: "modern", 3: "ancient"}
    name2label = {"background":0, "cartoon":1, "modern":2, "ancient":3}
    label2name_w = {0: "watching", 1: "unwatching"}
    name2label_w = {"watching":0, "unwatching":1}
    # p = "/Users/chenziwen/Downloads/CaptureIMGS/images"

    l = glob.glob("/Users/chenziwen/Downloads/CaptureIMGS/*")
    dl = ['/Users/chenziwen/Downloads/CaptureIMGS/background',
          '/Users/chenziwen/Downloads/CaptureIMGS/cartoon',
          '/Users/chenziwen/Downloads/CaptureIMGS/modern',
          '/Users/chenziwen/Downloads/CaptureIMGS/ancient']
    watchls = '/Users/chenziwen/Downloads/CaptureIMGS/1/观影中'
    watchlno = '/Users/chenziwen/Downloads/CaptureIMGS/1/观影否'
    watchd = '/Users/chenziwen/Downloads/CaptureIMGS/1'

    p = "/Users/chenziwen/Downloads/CaptureIMGS/background"
    rename(p)

    # syncfile2(dl, watchls, watchlno)
    # syncfile(dl, watchls, watchlno)
    # gen_txt(dl, watchd, name2label, name2label_w)

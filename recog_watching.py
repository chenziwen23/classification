# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/28 11:12
File Description:
    识别是否在观影

    结果如下：
loaded data   10.820189237594604
---------  begin ------------
---------  lr ------------
scores：0.8830260648442466

---------  ada ------------
scores：0.910680228862047
1.4066696166992188e-05    ----    0.03586292266845703

"""
import os
import random
import time

import numpy as np
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import AdaBoostClassifier
    c = time.time()
    x, y, xt, yt = prepare()
    print('loaded data  ', time.time() - c)

    print('---------  begin ------------ ')
    lr_clf = LogisticRegression()
    ada_clf = AdaBoostClassifier(n_estimators=200, random_state=0)
    lr_clf.fit(x, y)
    ada_clf.fit(x, y)

    xts = np.concatenate([x, xt], axis=0)
    yts = np.concatenate((y, yt), axis=0)

    y_label_new_predict = lr_clf.predict(xts)
    print('---------  lr ------------ ')
    s = time.time()
    print(np.sum(y_label_new_predict == yts) / len(yts))
    e = time.time() - s

    print('\n---------  ada ------------ ')
    s = time.time()
    y_label_new_predict = ada_clf.predict(xts)
    print(np.sum(y_label_new_predict == yts) / len(yts))

    print(e, '   ----   ', time.time() - s)


if __name__ == '__main__':
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/images"
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/background"
    src = "/Volumes/KINGSTON/test"
    dst = "/Users/chenziwen/Downloads/CaptureIMGS/images_mini"
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    # dst = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    main()

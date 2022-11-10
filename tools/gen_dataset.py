# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/25 16:13
File Description:

"""

import os
import glob
import random


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


if __name__ == '__main__':
    # dir2txt('/Users/chenziwen/Downloads/MFRD', txt_dir='./')
    # 融合test.csv 和 标签
    # import pandas as pd
    #
    # train = '/mnt/chenziwen/proserve/train.csv'
    # test = '/mnt/chenziwen/proserve/test.csv'
    # file2 = '/Users/chenziwen/Downloads/id2mac2label.csv'
    #
    # f_df = pd.read_csv(file2, index_col='mac')
    # f_df.columns
    # # x = f_df.loc['B01C0C5D8B8F']
    # # print(x.name, x.values,end=)
    #
    # # list of strings
    # lst = [['fav', 'tutor', 'coding', 'skills'],
    #        ['fa1', 'tuto2', 'codin2', 'skill2']]
    # df = pd.DataFrame(lst)
    # df.to_csv()
    # print(df)
    import time
    import numpy as np
    a = ['b01c0c5be7d6', 'b01c0c5bfe94', 'b01c0c5c5b0e', 'b01c0c5c76e9', 'b01c0c5c76e9']
    b = ['b01c0c5bfe94', 'B01c0c5c5b0e']

    s = time.time()
    print(list(set(a).intersection(set(b))))
    print(time.time() - s)

    s = time.time()
    print([j for j in a if j in b])
    print(time.time() - s)

    s = time.time()
    print(np.intersect1d(a, b))
    print(time.time() - s)
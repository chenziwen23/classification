# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/11/6 15:40
File Description:

"""
import os
import cv2
from io import BytesIO
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
        img_new.save(dis_img)
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
            # img_new.save(dis_img)


def gen_txt_dc(root='/mnt/chenziwen/Datasets/dc/train', path_label='/mnt/chenziwen/Datasets/dc/label.txt'):

    img_name_list = os.listdir(root)
    txt_content = []
    for img_n in img_name_list:
        if 'dog' in img_n:
            txt_content.append(f"{img_n},{1}\n")
        elif 'cat' in img_n:
            txt_content.append(f"{img_n},{0}\n")
        else:
            raise ValueError(f"The image({img_n}) couldn't recognized!")

    with open(path_label, 'w') as f:
        f.writelines(txt_content)
    print(f'path_label is written in {path_label}')


def show(name):
    root = '/Users/chenziwen/Downloads/CaptureIMGS/images_mini'
    file_p = os.path.join(root, name)
    img = Image.open(file_p).convert('L')
    img = img.filter(ImageFilter.FIND_EDGES)
    img.show('s')
    # img = cv2.imread(file_p)
    # new_img = cv2.Canny(img, 80, 160)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('asda ', new_img)
    # cv2.waitKey(0)


def main():
    # img_trans(src, dst)
    # gen_txt_dc()
    # show('00005180.jpg')
    show('00001167.jpg')
    # show('00005167.jpg')


if __name__ == '__main__':
    # src = "/Users/chenziwen/Downloads/CaptureIMGS/images"
    dst = "/Users/chenziwen/Downloads/CaptureIMGS/images_mini"
    src = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    # dst = "/Users/chenziwen/Downloads/CaptureIMGS/images/00000831.png"
    main()



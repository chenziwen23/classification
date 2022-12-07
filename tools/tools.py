# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/10/25 21:47
File Description:

"""
import math
import os
import glob
import random

import cv2
from concurrent.futures import ThreadPoolExecutor

def rename(p):
    # rename random string
    imgs_p = os.listdir(p)
    for i in range(len(imgs_p)):
        imgp = os.path.join(p, imgs_p[i])
        n = imgs_p[i].split('.')[0]
        if n == '':
            continue
        imgp_new = os.path.join(p, f"{int(i+2398):0>8d}.png")
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


# 提取出 name logo 以及 直播流
def println(url_slist, logo_dir):
    import re
    import requests
    from urllib.request import urlretrieve

    def download(file_path, picture_url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 			(KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE",
        }
        r = requests.get(picture_url, headers=headers)
        with open(file_path, 'wb') as f:
            f.write(r.content)

    response = requests.get(url_slist)
    text = response.text.split('#EXTINF:-1')
    for t in text:
        name = re.findall(r'tvg-name="(.+?)"', t)
        name = name[0] if len(name) != 0 else None

        logo = re.findall(r'tvg-logo="(.+?)"', t)
        logo = logo[0] if len(logo) != 0 else None

        if logo and name:
            if 'png' in logo:
                # print(name, " : ", logo)
                path = os.path.join(logo_dir, name + '.png')
                if not os.path.exists(path):
                    try:
                        urlretrieve(logo, path)
                    except:
                        print(name, " : ", logo)
                        download(path, logo)


################ ###################
dict_m = {'非自然死亡：第4话 为了谁而工作.mp4': 2, '人在囧途.mp4': 2, '西游记：第10集 三打白骨精.mp4': 3, '最爱：第7集 .mp4': 2, '大宅门：第1集 .mp4': 3,
              '铁齿铜牙纪晓岚4：第2集 .mp4': 3, '非自然死亡：第6话 不是朋友.mp4': 2, '请和废柴的我谈恋爱：第8话 .mp4': 2, '罗小黑战记.mp4': 1,
              '三国演义：第27集 三顾茅庐.mp4': 3,
              '大宅门：第30集 .mp4': 3, '老友记 第一季：第10集 猴子.mp4': 2, '铁齿铜牙纪晓岚4：第6集 .mp4': 3,
              '王牌对王牌 第五季：华晨宇极致宠粉隔空送“礼物” 史上最暖云录制圆梦“最美医护工作者”.mp4': 2, '康熙王朝：第30集 .mp4': 3, '康熙王朝：第1集 .mp4': 3,
              '西游记：第18集 扫塔辨奇冤.mp4': 3, '铁齿铜牙纪晓岚4：第1集 .mp4': 3, '康熙王朝：第13集 .mp4': 3, '请和废柴的我谈恋爱：第10话 .mp4': 2,
              '大宅门：第17集 .mp4': 3,
              '让子弹飞.mp4': 2, '铁齿铜牙纪晓岚4：第7集 .mp4': 3, '最爱：第2集 .mp4': 2, '铁齿铜牙纪晓岚4：第11集 .mp4': 3,
              '老友记 第一季：第1集 莫妮卡的新室友.mp4': 2,
              '四海.mp4': 2, '老友记 第一季：第15集 麻烦的家伙.mp4': 2, '非自然死亡：第9话 敌人的身影.mp4': 2, '康熙王朝：第17集 .mp4': 3, '碟中谍4.mp4': 2,
              '德雷尔一家 第一季：第3集 .mp4': 2, '功夫.mp4': 2, '哔哩哔哩向前冲：第12期 是兄弟，就来向前冲落水！.mp4': 2, '德雷尔一家 第一季：第1集 .mp4': 2,
              '爱的二八定律 第02集.mp4': 2, '老友记 第一季：第22集 肉麻元素.mp4': 2, '三国演义：第35集 苦肉计.mp4': 3, '非自然死亡：第2话 死志遗信.mp4': 2,
              '猫和老鼠：大电影.mp4': 1, '金蝉脱壳.mp4': 2, '三国演义：第23集 大破袁绍.mp4': 3, '康熙王朝：第22集 .mp4': 3, '乡村爱情 第1季：第10集 .mp4': 2,
              '哆啦A梦：大雄的新恐龙.mp4': 1, '大宅门：第9集 .mp4': 3, '金刚：骷髅岛.mp4': 2, '乡村爱情 第1季：第1集 .mp4': 2,
              '珍馐记：第10集 最美的风景，最真诚的心意.mp4': 3, '三国演义：第31集 智激周瑜.mp4': 3, '铁齿铜牙纪晓岚4：第23集 .mp4': 3, '西游记：第2集 官封弼马温.mp4': 3,
              '铁齿铜牙纪晓岚4：第27集 .mp4': 3, '西游记：第19集 误入小雷音.mp4': 3, '哔哩哔哩向前冲：第15期 复仇勇士卷土重来！.mp4': 2,
              '王牌对王牌 第五季：沈腾迷惑道具逆袭体育界 贾玲爆笑诠释吨位版张柏芝.mp4': 1, '凡人修仙传：第30话 魔道争锋9.mp4': 1, '我叫MT：归来：第2话 这冒险不会辜负我们的到来.mp4': 1,
              '凡人修仙传：第13话 凡人风起天南13.mp4': 1, '凡人修仙传：第19话 燕家堡之战2.mp4': 1, '我叫MT：归来：第3话 燃烧吧 征程开始的地方.mp4': 1, '元龙 第二季：第6话 宋嫣进城.mp4': 1,
              '我叫MT：归来：第1话 即使逆流时光我们仍会相逢.mp4': 1, '元龙 第二季：第1话 王胜进城.mp4': 1, '凡人修仙传：第18话 燕家堡之战1.mp4': 1}


# 将视频转换成图像，并以标签命名
def video2img(video_path, imgdir, lab, j=0):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(3)
    h = cap.get(4)
    fps = int(cap.get(5))
    fpsd = fps * 4

    ret = cap.isOpened()
    i = 0
    s = 0
    start_f = int(fps * 2.5 * 60)
    print(f"fps : {fps} ,  w : {w} ,  h : {h},   j : {j}  -  {video_path}")
    while ret:
        ret, frame = cap.read()
        i += 1
        if frame is not None:
            if i > start_f:
                if i % fpsd == 0:
                    s += 1
                    cv2.imwrite(os.path.join(imgdir, f'{lab}_' + f"{j:0>3d}{s:0>5d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cap.release()
    return


def traversal_videos(videos_dir, imgdir):
    j = 0
    for video in os.listdir(videos_dir):
        if video == ".DS_Store":
            continue
        video2img(os.path.join(videos_dir, video), imgdir, dict_m[video], j)
        j += 1


# image (720p) to below 640 (maximum)
def tool_resize(img_n, img_dir, new_dir, size_max):
    new_p = os.path.join(new_dir, img_n)
    if os.path.exists(new_p):
        return
    try:
        img = cv2.imread(os.path.join(img_dir, img_n))
        h, w = img.shape[:-1]
        ratio = min(size_max / h, size_max / w)
        new_h, new_w = int(ratio * h), int(ratio * w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_p, img)
        print(os.path.join(new_dir, img_n))
    except:
        print(os.path.join(new_dir, img_n),  '--------------  unsuccessful   --------')


def resize(img_dir, new_dir, size_max=640):
    img_list = os.listdir(img_dir)
    executor = ThreadPoolExecutor(16)
    for img_n in img_list:
        executor.submit(tool_resize, img_n, img_dir, new_dir, size_max)

    executor.shutdown(wait=True)


# label
def img_with_label(img_dir):
    from mmdet.apis import init_detector, inference_detector
    config_file = '/mnt/chenziwen/cv/yolov3_mobilenetv2_320_300e_coco.py'
    checkpoint_file = '/mnt/chenziwen/cv/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

    img_list = os.listdir(img_dir)
    for ip in img_list:
        # image = fr.load_image_file(img_dir + '/' + ip)
        # face_locations = fr.face_locations(image, model='cnn')
        if int(ip.split('_')[0]) == 1:
            continue
        results = inference_detector(model, img_dir + '/' + ip)[0]

        if len(results) != 0:
            scores = results[0][-1]
            if scores > 0.8:
                continue
        old_ip = '0_' + ip[:1] + ip[3:]
        os.rename(os.path.join(img_dir, ip), os.path.join(img_dir, old_ip))
        print(os.path.join(img_dir, ip),  '  -> ', os.path.join(img_dir, old_ip))


if __name__ == '__main__':
    label2name = {0: "background", 1: "cartoon", 2: "modern", 3: "ancient"}
    name2label = {"background":0, "cartoon":1, "modern":2, "ancient":3}
    label2name_w = {0: "watching", 1: "unwatching"}
    name2label_w = {"watching":0, "unwatching":1}
    # p = "/Users/chenziwen/Downloads/CaptureIMGS/images"


    watchls = '/Users/chenziwen/Downloads/CaptureIMGS/1/观影中'
    watchlno = '/Users/chenziwen/Downloads/CaptureIMGS/1/观影否'
    watchd = '/Users/chenziwen/Downloads/CaptureIMGS/1'

    # syncfile2(dl, watchls, watchlno)
    # syncfile(dl, watchls, watchlno)
    # gen_txt(dl, watchd, name2label, name2label_w)

    """
    url_slist = "https://raw.githubusercontent.com/YueChan/IPTV/main/IPTV.m3u"
    logo_dir = "/Users/chenziwen/Downloads/CaptureIMGS/tvlogo"
    println(url_slist, logo_dir)
    
    videos_dir = "/Users/chenziwen/Downloads/CaptureIMGS/movies"
    imgdir = "/Users/chenziwen/Downloads/CaptureIMGS/images"
    traversal_videos(videos_dir, imgdir)
    
    imgdir = "/Users/chenziwen/Downloads/CaptureIMGS/images"
    new_dir = "/Users/chenziwen/Downloads/CaptureIMGS/movie_imgs_640"
    resize(imgdir, new_dir)
    
    imgdir = "/mnt/chenziwen/Datasets/movie_imgs_640"
    img_with_label(imgdir)
    
    """


    'http://epg.51zmt.top:8000/tb1/gt/fenghuangzixun.png'




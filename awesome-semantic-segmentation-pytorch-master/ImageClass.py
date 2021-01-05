import os

from typing import List, Any

import numpy as np

import codecs

import json

from glob import glob

import cv2

import shutil

from sklearn.model_selection import train_test_split
#
#这个文件是把png和jpg生成对应voc格式数据集，并把相应文件写进txt里

#
# 1.标签路径

labelme_path = "datasets/own/CRACK500/traincrop/"# 原始labelme标注数据路径

saved_path = "./dataOUt/" # 保存路径

isUseTest = True  # 是否创建test集

# 2.创建要求文件夹

if not os.path.exists(saved_path + "Annotations"):

     os.makedirs(saved_path + "Annotations")

if not os.path.exists(saved_path + "JPEGImages/"):

    os.makedirs(saved_path + "JPEGImages/")

if not os.path.exists(saved_path + "ImageSets/Main/"):

    os.makedirs(saved_path + "ImageSets/Main/")


# 5.复制图片到 VOC2007/JPEGImages/下

image_files = glob(labelme_path + "*.jpg")

print("copy image files to VOC007/JPEGImages/")

for image in image_files:

    shutil.copy(image, saved_path + "JPEGImages/")


mask_files = glob(labelme_path + "*.png")

for image in mask_files:

    shutil.copy(image, saved_path + "Annotations/")

# filelist = os.listdir("./datasets/own/CRACK500/valdata/Annotations")
# total_num = len(filelist)  # 获取文件夹内所有文件个数
# i = 1  # 表示文件的命名是从1开始的
# for item in filelist:
#     if item.endswith('.png'):
#     # 初始的图片的格式为png格式的
#         src = os.path.join(os.path.abspath("./datasets/own/CRACK500/valdata/Annotations"), item)
#         dst = os.path.join(os.path.abspath("./datasets/own/CRACK500/valdata/Annotations"), item.replace("_mask", ""))
#         try:
#             os.rename(src, dst)
#             print('converting %s to %s ...' % (src, dst))
#             i = i + 1
#         except:
#             continue
#
# print("copy image files to VOC007/Annotations/")


# 6.split files for txt

txtsavepath = saved_path + "ImageSets/Main/"

ftrainval = open(txtsavepath + '/trainval.txt', 'w')

ftest = open(txtsavepath + '/test.txt', 'w')

ftrain = open(txtsavepath + '/train.txt', 'w')

fval = open(txtsavepath + '/val.txt', 'w')

total_files = glob("./dataOUt/Annotations/*.png")

total_files = [i.replace("\\", "/").split("/")[-1].split(".png")[0] for i in total_files]

trainval_files = []

test_files = []

if isUseTest:
    print(len(total_files))

    trainval_files, test_files = train_test_split(total_files, test_size=0.1, random_state=55)

else:

    trainval_files = total_files

for file in trainval_files:

    ftrainval.write(file + "\n")

# split

train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)

# train

for file in train_files:

    ftrain.write(file + "\n")

# val

for file in val_files:

    fval.write(file + "\n")

for file in test_files:

    print(file)

    ftest.write(file + "\n")

ftrainval.close()

ftrain.close()

fval.close()

ftest.close()
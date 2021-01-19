import os
import sys
import argparse
import torch
import time

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model
from core.data.dataloader import get_segmentation_dataset
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float
import scipy.misc

import torch.utils.data as data
from core.utils.distributed import *
import numpy as np


parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default= None,
                    help='path to the input picture')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
    print('Finished loading model!')

    if args.input_pic != None:
        image = Image.open(config.input_pic).convert('RGB')
        images = transform(image).unsqueeze(0).to(device)
        test(model, images, args.input_pic)
    else:
    # image transform
        test_dataset = get_segmentation_dataset(args.dataset, split='test', mode='test', transform=transform)
        test_sampler = make_data_sampler(test_dataset, True, False)
        test_batch_sampler = make_batch_data_sampler(test_sampler, images_per_batch=1)
        test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_sampler=test_batch_sampler,
                                    num_workers=4,
                                    pin_memory=True)
        for i, (image, target) in enumerate(test_loader):
            image = image.to(torch.device(device))
            test(model, image, ''.join(target))


def test(model, image, name):
    starttime = time.time()
    model.eval()
    with torch.no_grad():
        output = model(image)

    # endtime = time.time()
    #
    # fps = 1. / (endtime - starttime)
    # print("fps:", fps)
    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    file = os.path.splitext(os.path.split(name)[-1])[0]
    outname1 = file + '.jpg'#原图名
    outname2 = file + '.png'#mask文件名
    # mask.save(os.path.join(args.outdir, outname2))

    # image1 原图
    # image2 分割图
    image1 = Image.open(os.path.join('../datasets/own/VOC2012/JPEGImages', outname1))#打开原图
    #image2 = Image.open(os.path.join(args.outdir, outname2))


    image1 = image1.convert('RGBA')
    mask = mask.convert('RGBA')

    # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    image = Image.blend(image1, mask, 0.6)
    # # image.save("test.png")
    #
    # gam2 = exposure.adjust_gamma(img_as_float(image), 0.5)  # 调节图片亮度，0.5为变亮，1为不变，2为变暗
    # plt.imsave(os.path.join(args.outdir, outname2), image, cmap='plasma')  # 融合图片保存在../scripts/eval文件夹下
    image.save(os.path.join(args.outdir, outname2))
    endtime = time.time()

    fps = 1. / (endtime - starttime)
    print("fps:", fps)





if __name__ == '__main__':
    demo(args)

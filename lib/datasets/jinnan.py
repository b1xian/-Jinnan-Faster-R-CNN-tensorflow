from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

from lib.config import config as cfg
from lib.datasets.imdb import imdb
from datasets.voc_eval import voc_eval
import json
"""
# 加载json格式的ground-truth文件
f = open('{}train_no_poly.json'.format('../../data/jinnan2_round1_train_20190222/'), encoding='utf-8')
gt = json.load(f)
images = gt['images']
anns = gt['annotations']
# 类别加上背景
num_classes = len(gt['categories']) + 1

normal_file_path = os.path.join('../../data/jinnan2_round1_train_20190222', 'normal')
image_index = []
for file in os.listdir(normal_file_path):
    image_index.append(file[:file.index('.')])
restricted_file_path = os.path.join('../../data/jinnan2_round1_train_20190222', 'restricted')
for file in os.listdir(restricted_file_path):
    image_index.append(file[:file.index('.')])
print(len(image_index))


def _load_pascal_annotation(index, images, anns):
    # Load image and bounding boxes info from XML file in the PASCAL VOC
    # format.
    # 遍历ground-truth信息,找到这张图片的所有ground-truth
    gts = []
    for img_info in images:
        if img_info['file_name'] == (str(index) + '.jpg'):
            # 取anno信息
            img_id = img_info['id']
            for ann in anns:
                # 一张图片可以对应多个annotation
                if ann['image_id'] == img_id:
                    gts.append(ann)
    num_objs = len(gts)
    # ground-truth
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    # 分类
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    for ix, gt in enumerate(gts):
        # 类别id
        cls = gt['category_id']
        # bbox [x, y, width, height]
        bbox = gt['bbox']
        x = float(bbox[0])
        y = float(bbox[1])
        h = float(bbox[2])
        w = float(bbox[3])
        gt_classes[ix] = cls
        # bbox 需要[x1, y1, x2, y2]
        boxes[ix, :] = [x, y, x + w, y + h]
        seg_areas[ix] = h * w
        overlaps[ix, cls] = 1.0
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}



gt_roidb = [_load_pascal_annotation(index, images, anns) for index in image_index]
print(gt_roidb[2541])
print(gt_roidb[2541]['gt_overlaps'])
print(gt_roidb[2541]['gt_classes'])
print(gt_roidb[2541]['seg_areas'])
print(gt_roidb[2541]['boxes'])
"""

image_path = os.path.join("../../data/jinnan2_round1_train_20190222", 'normal',
                                  "190102_152618_00148882.jpg")
# image_path = os.path.join("../../data/jinnan2_round1_train_20190222", 'restricted',
#                                   "190102_152618_00148882.jpg")
if os.path.exists(image_path):
    print(" exists")
else:
    print("not exists")

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
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


# 继承imdb类
class jinnan(imdb):
    def __init__(self, image_set='jinnan', data_path=None):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.FLAGS2["data_dir"], data_path)
        self._devkit_path = os.path.join(cfg.FLAGS2["data_dir"], str(data_path+'_devkit\\'))
        self._classes = ('__background__',  # always index 0
                         "iron-lighter", "black-lighter", "knive", "battery", "scissor")
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

    # 通过图片名称(索引)，获取图片路径
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    # 通过图片名称(索引)，获取图片路径
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path, 'normal',
        #                           index + self._image_ext)
        # if not os.path.exists(image_path):
        image_path = os.path.join(self._data_path, 'restricted',
                              index + self._image_ext)
        if not os.path.exists(image_path):
            image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # 封装所有图片名
    def _load_image_set_index(self):
        image_index = []
        # normal_file_path = os.path.join(self._data_path, '/normal')
        # normal_file_path = self._data_path+'/normal'
        # for file in os.listdir(normal_file_path):
        #     image_index.append(file[:file.index('.')])

        restricted_file_path = os.path.join(self._data_path, 'restricted')
        if not os.path.exists(restricted_file_path):
            restricted_file_path = self._data_path
        for file in os.listdir(restricted_file_path):
            image_index.append(file[:file.index('.')])
        return image_index

    # 封装ground-truth数据
    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        # 加载json格式的ground-truth文件
        f = open('{}\\train_no_poly.json'.format(self._data_path), encoding='utf-8')
        gt = json.load(f)
        images = gt['images']
        anns = gt['annotations']
        # 类别加上背景
        num_classes = len(gt['categories']) + 1
        # 获取ground-truth集合，通过图片名，取每个图片的ground-truth信息
        gt_roidb = [self._load_pascal_annotation(index, images, anns)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    # 暂时没用，不重写了
    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb
    # 暂时没用，不重写了
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    # 加载ground-truth数据
    def _load_pascal_annotation(self, index, images, anns):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
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
        num_gts = len(gts)
        # ground-truth
        boxes = np.zeros((num_gts, 4), dtype=np.uint16)
        # 分类
        gt_classes = np.zeros((num_gts), dtype=np.int32)
        overlaps = np.zeros((num_gts, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_gts), dtype=np.float32)
        for ix, gt in enumerate(gts):
            # 类别id
            cls = gt['category_id']
            # 数据集bbox [x, y, width, height]
            bbox = gt['bbox']
            x = float(bbox[0])
            y = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
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

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_jinnan_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            filename)
        return path

    def _write_jinnan_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Jinnan results file'.format(cls))
            filename = self._get_jinnan_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = self._devkit_path + '\\VOC' + self._year + '\\Annotations\\' + '{:s}.xml'
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_jinnan_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.FLAGS2["root_dir"], 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    # 校验预测结果
    def evaluate_detections(self, all_boxes, output_dir):
        # all detections are collected into:
        #  all_boxes[cls][image] = N x 5 array of detections in
        #  (x1, y1, x2, y2, score)
        self._write_jinnan_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_jinnan_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc

    d = jinnan(data_path='jinnan2_round1_train_20190222\\')
    d = jinnan(data_path='jinnan2_round1_test_20190222\\')
    res = d.roidb
    from IPython import embed

    embed()

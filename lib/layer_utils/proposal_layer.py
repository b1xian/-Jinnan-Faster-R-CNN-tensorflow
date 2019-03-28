# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        # 预处理，非极大值抑制前，保留12000个bbox
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n
        # 非极大值抑制后，选取2000个bbox
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n
        # 非极大值抑制阈值0.7，与最大概率的bbox IoU超过0.7的bbox会被丢弃
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh
    else:
        # 预处理，非极大值抑制前，6000个bbox
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        # 非极大值抑制后，选取300个bbox
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        # nms阈值0.7
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0]
    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    # 按照分类得分，取前pre_nms_topN个bbox(12000/6000)
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 做nms
    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # nms后保留post_nms_topN个bbox(2000/300)
    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    # 返回bbox以及分类得分
    return blob, scores

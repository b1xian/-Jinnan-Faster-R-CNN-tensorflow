# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim

import lib.config.config as cfg
from lib.nets.network import Network

class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('vgg_16', 'vgg_16'):

            # select initializer  使用truncated的正太分布来初始化w
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head 构建基础特征提取网络，VGG16，使用conv5_3及之前的网络层
            net = self.build_head(is_training)

            # Build rpn 构建rpn网络,对所有anchor进行二分类和回归
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            # Build proposals  候选框选取,roi标签及回归值设置
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_head(self, is_training):

        # Main network
        # Layer  1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # Build anchor component
        # 初始化anchor
        self._anchor_component()

        # Create RPN Layer
        # 首先用一个3*3的卷积，目的是为了是一个点尽可能表示周边的信息
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
        print('rpn', rpn)

        self._act_summaries.append(rpn)
        # anchor box的二分类输出,使用2*k个1*1卷积核
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # Change it so that the score has 2 as its channel size
        # reshape，每个anchor box输出2个值用于分类
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        # 使用softmax做二分类
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        # anchor box的二分类输出
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        # anchor box的回归输出
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):


        if is_training:
            # 如果是训练，需要选取roi
            # 初步的过滤，使用nms保留约2000个bbox
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            # 筛选过界的bbox，为bbox设置label,并设置inside和outside权重
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            # 预测
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        # 将不同大小的roi处理为相同大小，此处为了省时间，作者直接用了crop和resize
        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        print('pool5 after resize:', pool5)
        # 后面用的是全连接，使用全卷积会有更好的效果
        pool5_flat = slim.flatten(pool5, scope='flatten')
        print('pool5_flat:', pool5_flat)

        # Fully connected layers
        # 两层全连接
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        # Scores and predictions,输出所有类别的score
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        print("cls_score:", cls_score)
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        # 输出回归值
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')
        print("bbox_prediction:", bbox_prediction)
        return cls_score, cls_prob, bbox_prediction

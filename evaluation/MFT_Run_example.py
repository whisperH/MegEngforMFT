import os
import numpy as np
import json
import cv2

import argparse
import bisect
import copy
import time
import json
import math
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
from typing import List
import matplotlib.pyplot as plt

import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.optimizer import SGD
from megengine import Tensor, tensor
from megengine.module.normalization import GroupNorm, InstanceNorm, LayerNorm
from functools import partial

import os

import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

import megengine.functional as F
import megengine.module as M

race_path = 'preliminary' # 'preliminary' or 'intermediary'
DATA_PATH = os.path.join('/home/huangjinze/code/data/MFTChallenge/preliminary')

SPLITS = ['test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True


def meshgrid(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y

def create_anchor_grid(featmap_size, offsets, stride, device):
    step_x, step_y = featmap_size
    shift = offsets * stride

    grid_x = F.arange(shift, step_x * stride + shift, step=stride, device=device)
    grid_y = F.arange(shift, step_y * stride + shift, step=stride, device=device)
    grids_x, grids_y = meshgrid(grid_y, grid_x)
    return grids_x.reshape(-1), grids_y.reshape(-1)

class BaseAnchorGenerator(metaclass=ABCMeta):
    """base class for anchor generator.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def anchor_dim(self):
        pass

    @abstractmethod
    def generate_anchors_by_features(self, sizes, device) -> List[Tensor]:
        pass

    def __call__(self, featmaps):
        feat_sizes = [fmap.shape[-2:] for fmap in featmaps]
        return self.generate_anchors_by_features(feat_sizes, featmaps[0].device)


class AnchorBoxGenerator(BaseAnchorGenerator):

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        anchor_scales: list = [[32], [64], [128], [256], [512]],
        anchor_ratios: list = [[0.5, 1, 2]],
        strides: list = [4, 8, 16, 32, 64],
        offset: float = 0.5,
    ):
        super().__init__()
        self.anchor_scales = np.array(anchor_scales, dtype=np.float32)
        self.anchor_ratios = np.array(anchor_ratios, dtype=np.float32)
        self.strides = strides
        self.offset = offset
        self.num_features = len(strides)

        self.base_anchors = self._different_level_anchors(anchor_scales, anchor_ratios)

    @property
    def anchor_dim(self):
        return 4

    def _different_level_anchors(self, scales, ratios):
        if len(scales) == 1:
            scales *= self.num_features
        assert len(scales) == self.num_features

        if len(ratios) == 1:
            ratios *= self.num_features
        assert len(ratios) == self.num_features
        return [
            tensor(self.generate_base_anchors(scale, ratio))
            for scale, ratio in zip(scales, ratios)
        ]

    def generate_base_anchors(self, scales, ratios):
        base_anchors = []
        areas = [s ** 2.0 for s in scales]
        for area in areas:
            for ratio in ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                # center-based anchor
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                base_anchors.append([x0, y0, x1, y1])
        return base_anchors

    def generate_anchors_by_features(self, sizes, device):
        all_anchors = []
        assert len(sizes) == self.num_features, (
            "input features expected {}, got {}".format(self.num_features, len(sizes))
        )
        for size, stride, base_anchor in zip(sizes, self.strides, self.base_anchors):
            grid_x, grid_y = create_anchor_grid(size, self.offset, stride, device)
            grids = F.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
            all_anchors.append(
                (F.expand_dims(grids, axis=1) + F.expand_dims(base_anchor, axis=0)).reshape(-1, 4)
            )
        return all_anchors

class BoxCoderBase(metaclass=ABCMeta):
    """Boxcoder class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def encode(self) -> Tensor:
        pass

    @abstractmethod
    def decode(self) -> Tensor:
        pass


class BoxCoder(BoxCoderBase, metaclass=ABCMeta):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        reg_mean=[0.0, 0.0, 0.0, 0.0],
        reg_std=[1.0, 1.0, 1.0, 1.0],
    ):
        """
        Args:
            reg_mean(np.ndarray): [x0_mean, x1_mean, y0_mean, y1_mean] or None
            reg_std(np.ndarray):  [x0_std, x1_std, y0_std, y1_std] or None

        """
        self.reg_mean = np.array(reg_mean, dtype=np.float32)[None, :]
        self.reg_std = np.array(reg_std, dtype=np.float32)[None, :]
        super().__init__()

    @staticmethod
    def _box_ltrb_to_cs_opr(bbox, addaxis=None):
        """ transform the left-top right-bottom encoding bounding boxes
        to center and size encodings"""
        bbox_width = bbox[:, 2] - bbox[:, 0]
        bbox_height = bbox[:, 3] - bbox[:, 1]
        bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
        bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
        if addaxis is None:
            return bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y
        else:
            return (
                F.expand_dims(bbox_width, addaxis),
                F.expand_dims(bbox_height, addaxis),
                F.expand_dims(bbox_ctr_x, addaxis),
                F.expand_dims(bbox_ctr_y, addaxis),
            )

    def encode(self, bbox: Tensor, gt: Tensor) -> Tensor:
        bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y = self._box_ltrb_to_cs_opr(bbox)
        gt_width, gt_height, gt_ctr_x, gt_ctr_y = self._box_ltrb_to_cs_opr(gt)

        target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
        target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
        target_dw = F.log(gt_width / bbox_width)
        target_dh = F.log(gt_height / bbox_height)
        target = F.stack([target_dx, target_dy, target_dw, target_dh], axis=1)

        target -= self.reg_mean
        target /= self.reg_std
        return target

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        deltas *= self.reg_std
        deltas += self.reg_mean

        (
            anchor_width,
            anchor_height,
            anchor_ctr_x,
            anchor_ctr_y,
        ) = self._box_ltrb_to_cs_opr(anchors, 1)
        pred_ctr_x = anchor_ctr_x + deltas[:, 0::4] * anchor_width
        pred_ctr_y = anchor_ctr_y + deltas[:, 1::4] * anchor_height
        pred_width = anchor_width * F.exp(deltas[:, 2::4])
        pred_height = anchor_height * F.exp(deltas[:, 3::4])

        pred_x1 = pred_ctr_x - 0.5 * pred_width
        pred_y1 = pred_ctr_y - 0.5 * pred_height
        pred_x2 = pred_ctr_x + 0.5 * pred_width
        pred_y2 = pred_ctr_y + 0.5 * pred_height

        pred_box = F.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=2)
        pred_box = pred_box.reshape(pred_box.shape[0], -1)

        return pred_box


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Conv2d(M.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm):
    """
    Args:
        norm (str): currently support "BN", "SyncBN", "FrozenBN", "GN", "LN" and "IN"

    Returns:
        M.Module or None: the normalization layer
    """
    if norm is None:
        return None
    norm = {
        "BN": M.BatchNorm2d,
        "SyncBN": M.SyncBatchNorm,
        "FrozenBN": partial(M.BatchNorm2d, freeze=True),
        "GN": GroupNorm,
        "LN": LayerNorm,
        "IN": InstanceNorm,
    }[norm]
    return norm
class FPN(M.Module):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        bottom_up: M.Module,
        in_features: List[str],
        out_channels: int = 256,
        norm: str = None,
        top_block: M.Module = None,
        strides: List[int] = [8, 16, 32],
        channels: List[int] = [512, 1024, 2048],
    ):

        super(FPN, self).__init__()

        in_strides = strides
        in_channels = channels
        norm = get_norm(norm)

        use_bias = norm is None
        self.lateral_convs = list()
        self.output_convs = list()

        for idx, in_channels in enumerate(in_channels):
            lateral_norm = None if norm is None else norm(out_channels)
            output_norm = None if norm is None else norm(out_channels)

            lateral_conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm,
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            M.init.msra_normal_(lateral_conv.weight, mode="fan_in")
            M.init.msra_normal_(output_conv.weight, mode="fan_in")

            if use_bias:
                M.init.fill_(lateral_conv.bias, 0)
                M.init.fill_(output_conv.bias, 0)

            stage = int(math.log2(in_strides[idx]))

            setattr(self, "fpn_lateral{}".format(stage), lateral_conv)
            setattr(self, "fpn_output{}".format(stage), output_conv)
            self.lateral_convs.insert(0, lateral_conv)
            self.output_convs.insert(0, output_conv)

        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up

        # follow the common practices, FPN features are named to "p<stage>",
        # like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in in_strides
        }

        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(sorted(self._out_feature_strides.keys()))
        self._out_feature_channels = {k: out_channels for k in self._out_features}

    def forward(self, x):
        bottom_up_features = self.bottom_up.extract_features(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]

        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.nn.interpolate(
                prev_features, features.shape[2:], mode="BILINEAR"
            )
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(
                self.top_block.in_feature, None
            )
            if top_block_in_feature is None:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))

        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

class LastLevelP6P7(M.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels: int, out_channels: int, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        if in_feature == "p5":
            assert in_channels == out_channels
        self.in_feature = in_feature
        self.p6 = M.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = M.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

class BoxHead(M.Module):
    """
    The head used when anchor boxes are adopted for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        in_channels = input_shape[0].channels
        num_classes = cfg.num_classes
        num_convs = 4
        prior_prob = cfg.cls_prior_prob
        num_anchors = [
            len(cfg.anchor_scales[i]) * len(cfg.anchor_ratios[i])
            for i in range(len(input_shape))
        ]

        assert (
            len(set(num_anchors)) == 1
        ), "not support different number of anchors between levels"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(M.ReLU())
            bbox_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(M.ReLU())

        self.cls_subnet = M.Sequential(*cls_subnet)
        self.bbox_subnet = M.Sequential(*bbox_subnet)
        self.cls_score = M.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = M.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, M.Conv2d):
                    M.init.normal_(layer.weight, mean=0, std=0.01)
                    M.init.fill_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        M.init.fill_(self.cls_score.bias, bias_value)

    def forward(self, features: List[Tensor]):
        logits, offsets = [], []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            offsets.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, offsets


class Matcher:

    def __init__(self, thresholds, labels, allow_low_quality_matches=False):
        assert len(thresholds) + 1 == len(labels), "thresholds and labels are not matched"
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        thresholds.append(float("inf"))
        thresholds.insert(0, -float("inf"))

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, matrix):
        """
        matrix(tensor): A two dim tensor with shape of (N, M). N is number of GT-boxes,
            while M is the number of anchors in detection.
        """
        assert len(matrix.shape) == 2
        max_scores = matrix.max(axis=0)
        match_indices = F.argmax(matrix, axis=0)

        # default ignore label: -1
        labels = F.full_like(match_indices, -1)

        for label, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            mask = (max_scores >= low) & (max_scores < high)
            labels[mask] = label

        if self.allow_low_quality_matches:
            mask = (matrix == F.max(matrix, axis=1, keepdims=True)).sum(axis=0) > 0
            labels[mask] = 1

        return match_indices, labels
def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 0,
) -> Tensor:
    scores = F.sigmoid(logits)
    loss = -(targets * F.logsigmoid(logits) + (1 - targets) * F.logsigmoid(-logits))
    if gamma != 0:
        loss *= (targets * (1 - scores) + (1 - targets) * scores) ** gamma
    if alpha >= 0:
        loss *= targets * alpha + (1 - targets) * (1 - alpha)
    return loss


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:

    x = pred - target
    abs_x = F.abs(x)
    if beta < 1e-5:
        loss = abs_x
    else:
        in_loss = 0.5 * x ** 2 / beta
        out_loss = abs_x - 0.5 * beta
        loss = F.where(abs_x < beta, in_loss, out_loss)
    return loss

def get_clipped_boxes(boxes, hw):
    """ Clip the boxes into the image region."""
    # x1 >=0
    box_x1 = F.clip(boxes[:, 0::4], lower=0, upper=hw[1])
    # y1 >=0
    box_y1 = F.clip(boxes[:, 1::4], lower=0, upper=hw[0])
    # x2 < im_info[1]
    box_x2 = F.clip(boxes[:, 2::4], lower=0, upper=hw[1])
    # y2 < im_info[0]
    box_y2 = F.clip(boxes[:, 3::4], lower=0, upper=hw[0])

    clip_box = F.concat([box_x1, box_y1, box_x2, box_y2], axis=1)

    return clip_box

def get_iou(boxes1: Tensor, boxes2: Tensor, return_ioa=False) -> Tensor:

    b_box1 = F.expand_dims(boxes1, axis=1)
    b_box2 = F.expand_dims(boxes2, axis=0)

    iw = F.minimum(b_box1[:, :, 2], b_box2[:, :, 2]) - F.maximum(
        b_box1[:, :, 0], b_box2[:, :, 0]
    )
    ih = F.minimum(b_box1[:, :, 3], b_box2[:, :, 3]) - F.maximum(
        b_box1[:, :, 1], b_box2[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_box2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = F.expand_dims(area_box1, axis=1) + F.expand_dims(area_box2, axis=0) - inter
    overlaps = F.maximum(inter / union, 0)

    if return_ioa:
        ioa = F.maximum(inter / area_box1, 0)
        return overlaps, ioa

    return overlaps

def get_padded_tensor(
    array: Tensor, multiple_number: int = 32, pad_value: float = 0
) -> Tensor:
    batch, chl, t_height, t_width = array.shape
    padded_height = (
        (t_height + multiple_number - 1) // multiple_number * multiple_number
    )
    padded_width = (t_width + multiple_number - 1) // multiple_number * multiple_number

    padded_array = F.full(
        (batch, chl, padded_height, padded_width), pad_value, dtype=array.dtype
    )

    ndim = array.ndim
    if ndim == 4:
        padded_array[:, :, :t_height, :t_width] = array
    elif ndim == 3:
        padded_array[:, :t_height, :t_width] = array
    else:
        raise Exception("Not supported tensor dim: %d" % ndim)
    return padded_array

class RetinaNet(M.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.anchor_generator = AnchorBoxGenerator(
            anchor_scales=self.cfg.anchor_scales,
            anchor_ratios=self.cfg.anchor_ratios,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.box_coder = BoxCoder(cfg.reg_mean, cfg.reg_std)

        self.in_features = cfg.in_features

        # ----------------------- build backbone ------------------------ #
        bottom_up = mge.hub.load('megengine/models', 'resnet50', pretrained=True)
        del bottom_up.fc

        # ----------------------- build FPN ----------------------------- #
        self.backbone = FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=LastLevelP6P7(
                cfg.fpn_top_in_channel, cfg.fpn_out_channels, cfg.fpn_top_in_feature
            ),
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build RetinaNet Head ------------------ #
        self.head = BoxHead(cfg, feature_shapes)

        self.matcher = Matcher(
            cfg.match_thresholds, cfg.match_labels, cfg.match_allow_low_quality
        )

    def preprocess_image(self, image):
        padded_image = get_padded_tensor(image, 32, 0.0)
        normed_image = (
            padded_image
            - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
        ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
        return normed_image

    def forward(self, image, im_info, gt_boxes=None):
        image = self.preprocess_image(image)
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets = self.head(features)

        box_logits_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_anchors = F.concat(anchors_list, axis=0)

        if self.training:
            gt_labels, gt_offsets = self.get_ground_truth(
                all_level_anchors, gt_boxes, im_info[:, 4].astype(np.int32),
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            loss_cls = sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(num_fg, 1)

            loss_bbox = smooth_l1_loss(
                all_level_box_offsets[fg_mask],
                gt_offsets[fg_mask],
                beta=self.cfg.smooth_l1_beta,
            ).sum() / F.maximum(num_fg, 1) * self.cfg.loss_bbox_weight

            total = loss_cls + loss_bbox
            loss_dict = {
                "total_loss": total,
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert image.shape[0] == 1

            pred_boxes = self.box_coder.decode(
                all_level_anchors, all_level_box_offsets[0]
            )
            pred_boxes = pred_boxes.reshape(-1, 4)

            scale_w = im_info[0, 1] / im_info[0, 3]
            scale_h = im_info[0, 0] / im_info[0, 2]
            pred_boxes = pred_boxes / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_boxes = get_clipped_boxes(
                pred_boxes, im_info[0, 2:4]
            ).reshape(-1, 4)
            pred_score = F.sigmoid(all_level_box_logits)[0]
            return pred_score, clipped_boxes

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_num_gts):
        labels_list = []
        offsets_list = []

        for bid in range(batched_gt_boxes.shape[0]):
            gt_boxes = batched_gt_boxes[bid, :batched_num_gts[bid]]

            overlaps = get_iou(gt_boxes[:, :4], anchors)
            match_indices, labels = self.matcher(overlaps)
            gt_boxes_matched = gt_boxes[match_indices]

            fg_mask = labels == 1
            labels[fg_mask] = gt_boxes_matched[fg_mask, 4].astype(np.int32)
            offsets = self.box_coder.encode(anchors, gt_boxes_matched[:, :4])

            labels_list.append(labels)
            offsets_list.append(offsets)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
        )

class RetinaNetConfig:
    # pylint: disable=too-many-statements
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.backbone_norm = "FrozenBN"
        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = ["res3", "res4", "res5"]
        self.fpn_in_strides = [8, 16, 32]
        self.fpn_in_channels = [512, 1024, 2048]
        self.fpn_out_channels = 256
        self.fpn_top_in_feature = "res5"
        self.fpn_top_in_channel = 2048

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            ann_file=race_path+"_annotations/train_half.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            ann_file=race_path+"_annotations/val_half.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 1
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- net cfg ------------------------- #
        self.stride = [8, 16, 32, 64, 128]
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]
        self.reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [
            [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]
        ]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.match_thresholds = [0.4, 0.5]
        self.match_labels = [0, -1, 1]
        self.match_allow_low_quality = True
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.smooth_l1_beta = 0  # use L1 loss
        self.loss_bbox_weight = 1.0
        self.num_losses = 3

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333

        self.basic_lr = 0.01 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 54
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [42, 50]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5

from megengine.data.dataset.vision.meta_vision import VisionDataset

def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    return True


class FishCOCO(VisionDataset):
    r"""
    `MS COCO <http://cocodataset.org/#home>`_ Dataset.
    """

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        # TODO: need to check
        "info",
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, *, order=None
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            # for saving memory
            if "license" in img:
                del img["license"]
            if "coco_url" in img:
                del img["coco_url"]
            if "date_captured" in img:
                del img["date_captured"]
            if "flickr_url" in img:
                del img["flickr_url"]
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "polygons" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                image = cv2.imread(file_name, cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = [
                    self.json_category_id_to_contiguous_id[c] for c in boxes_category
                ]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["id"]]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "fish",
    )

    classes_originID = {
        "fish": 1
    }


from megengine.data import Collator, MapSampler, RandomSampler


class DetectionPadCollator(Collator):
    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def apply(self, inputs):
        """
        assume order = ["image", "boxes", "boxes_category", "info"]
        """
        batch_data = defaultdict(list)

        for image, boxes, boxes_category, info in inputs:
            batch_data["data"].append(image.astype(np.float32))
            batch_data["gt_boxes"].append(
                np.concatenate([boxes, boxes_category[:, np.newaxis]], axis=1).astype(
                    np.float32
                )
            )

            _, current_height, current_width = image.shape
            assert len(boxes) == len(boxes_category)
            num_instances = len(boxes)
            info = [
                current_height,
                current_width,
                info[0],
                info[1],
                num_instances,
            ]
            batch_data["im_info"].append(np.array(info, dtype=np.float32))

        for key, value in batch_data.items():
            pad_shape = list(max(s) for s in zip(*[x.shape for x in value]))
            pad_value = [
                np.pad(
                    v,
                    self._get_padding(v.shape, pad_shape),
                    constant_values=self.pad_value,
                )
                for v in value
            ]
            batch_data[key] = np.ascontiguousarray(pad_value)

        return batch_data

    def _get_padding(self, original_shape, target_shape):
        assert len(original_shape) == len(target_shape)
        shape = []
        for o, t in zip(original_shape, target_shape):
            shape.append((0, t - o))
        return tuple(shape)


class GroupedRandomSampler(RandomSampler):
    def __init__(
            self,
            dataset,
            batch_size,
            group_ids,
            indices=None,
            world_size=None,
            rank=None,
            seed=None,
    ):
        super().__init__(dataset, batch_size, False, indices, world_size, rank, seed)
        self.group_ids = group_ids
        assert len(group_ids) == len(dataset)
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def batch(self):
        indices = list(self.sample())
        if self.world_size > 1:
            indices = self.scatter(indices)

        batch_index = []
        for ind in indices:
            group_id = self.group_ids[ind]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(ind)
            if len(group_buffer) == self.batch_size:
                batch_index.append(group_buffer)
                self.buffer_per_group[group_id] = []

        return iter(batch_index)

    def __len__(self):
        raise NotImplementedError("len() of GroupedRandomSampler is not well-defined.")



def adjust_learning_rate(optimizer, epoch, step, cfg, args):
    base_lr = (
            cfg.basic_lr * args.batch_size * (
            cfg.lr_decay_rate
            ** bisect.bisect_right(cfg.lr_decay_stages, epoch)
    )
    )
    # Warm up
    lr_factor = 1.0
    if epoch == 0 and step < cfg.warm_iters:
        lr_factor = (step + 1.0) / cfg.warm_iters
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * lr_factor

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, record_len=1):
        self.record_len = record_len
        self.reset()

    def reset(self):
        self.sum = [0 for i in range(self.record_len)]
        self.cnt = 0

    def update(self, val):
        self.sum = [s + v for s, v in zip(self.sum, val)]
        self.cnt += 1

    def average(self):
        return [s / self.cnt for s in self.sum]


def train_one_epoch(model, data_queue, opt, gm, epoch, args):
    def train_func(image, im_info, gt_boxes):
        with gm:
            loss_dict = model(image=image, im_info=im_info, gt_boxes=gt_boxes)
            gm.backward(loss_dict["total_loss"])
            loss_list = list(loss_dict.values())
        opt.step().clear_grad()
        return loss_list

    meter = AverageMeter(record_len=model.cfg.num_losses)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())
    for step in range(tot_step):
        adjust_learning_rate(opt, epoch, step, model.cfg, args)

        data_tik = time.time()
        mini_batch = next(data_queue)
        data_tok = time.time()

        tik = time.time()
        loss_list = train_func(
            image=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"])
        )
        tok = time.time()
        time_meter.update([tok - tik, data_tok - data_tik])

        if dist.get_rank() == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(
                ["{}:%f".format(loss) for loss in model.cfg.losses_keys]
            )
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in loss_list])
            if step % log_interval == 0:
                logger.info(
                    log_info_str,
                    epoch,
                    step,
                    tot_step,
                    opt.param_groups[0]["lr"],
                    *meter.average(),
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def build_dataset(dataset_dir, cfg):
    data_cfg = copy.deepcopy(cfg.train_dataset)

    data_cfg["root"] = os.path.join(dataset_dir, 'coco')

    if "ann_file" in data_cfg:
        data_cfg["ann_file"] = os.path.join(data_cfg["root"], data_cfg["ann_file"])

    data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]
    return FishCOCO(
        data_cfg["root"],
        data_cfg["ann_file"],
        data_cfg["remove_images_without_annotations"],
        order=data_cfg["order"]
    )


def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(batch_size, dataset_dir, cfg):
    train_dataset = build_dataset(dataset_dir, cfg)
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=2,
    )
    return train_dataloader

class FileArgs:
    def __init__(self):
        self.devices = 0
        self.batch_size = 2
        self.dataset_dir = 'workspace/data'
        self.log_dir = 'workspace/log_of_retinanet_res50_coco_3x_800size'

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box. [x1, y1, x2, y2]
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.kf.F是状态变换模型
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        # self.kf.H是观测函数
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
        )
        # self.kf.R为测量噪声矩阵
        self.kf.R[2:, 2:] *= 10.
        # self.kf.P为协方差矩阵
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # self.kf.Q为过程噪声矩阵
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # 跟踪器数量为0则直接构造结果。
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections[:, :4], trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框估计值。
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

import random
def py_cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep

class DetEvaluator:
    def __init__(self, model):
        # @trace(symbolic=True)
        def pred_func(image, im_info):
            return model(image, im_info)

        self.model = model
        self.pred_func = pred_func

    @staticmethod
    def get_hw_by_short_size(im_height, im_width, short_size, max_size):
        im_size_min = np.min([im_height, im_width])
        im_size_max = np.max([im_height, im_width])
        scale = (short_size + 0.0) / im_size_min
        if scale * im_size_max > max_size:
            scale = (max_size + 0.0) / im_size_max

        resized_height, resized_width = (
            int(round(im_height * scale)),
            int(round(im_width * scale)),
        )
        return resized_height, resized_width

    @staticmethod
    def process_inputs(img, short_size, max_size, flip=False):
        original_height, original_width, _ = img.shape
        resized_height, resized_width = DetEvaluator.get_hw_by_short_size(
            original_height, original_width, short_size, max_size
        )
        resized_img = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR,
        )
        resized_img = cv2.flip(resized_img, 1) if flip else resized_img
        trans_img = np.ascontiguousarray(
            resized_img.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32
        )
        im_info = np.array(
            [(resized_height, resized_width, original_height, original_width)],
            dtype=np.float32,
        )
        return trans_img, im_info

    def predict(self, **inputs):
        box_cls, box_delta = self.pred_func(**inputs)
        # box_cls, box_delta = self.model(**inputs)
        box_cls, box_delta = box_cls.numpy(), box_delta.numpy()
        dtboxes_all = list()
        all_inds = np.where(box_cls > self.model.cfg.test_cls_threshold)

        for c in range(self.model.cfg.num_classes):
            inds = np.where(all_inds[1] == c)[0]
            inds = all_inds[0][inds]
            scores = box_cls[inds, c]
            if self.model.cfg.class_aware_box:
                bboxes = box_delta[inds, c, :]
            else:
                bboxes = box_delta[inds, :]

            dtboxes = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)

            if dtboxes.size > 0:
                if self.model.cfg.test_nms == -1:
                    keep = dtboxes[:, 4].argsort()[::-1]
                else:
                    assert 0 < self.model.cfg.test_nms <= 1.0
                    keep = py_cpu_nms(dtboxes, self.model.cfg.test_nms)
                dtboxes = np.hstack(
                    (dtboxes[keep], np.full((len(keep), 1), c, dtype=np.float32))
                ).astype(np.float32)
                dtboxes_all.extend(dtboxes)

        if len(dtboxes_all) > self.model.cfg.test_max_boxes_per_image:
            dtboxes_all = sorted(dtboxes_all, reverse=True, key=lambda i: i[4])[
                :self.model.cfg.test_max_boxes_per_image
            ]

        dtboxes_all = np.array(dtboxes_all, dtype=np.float)
        return dtboxes_all


class FishTracking(object):
    def __init__(self,
                 detector_model, detector_model_config, detector_weight,
                 tracker_model, tracker_weight,
                 fish_num=20, showRes=False
                 ):
        self.fish_num = fish_num

        self.detector_model = detector_model
        self.detector_model_config = detector_model_config
        self.detector_weight = detector_weight

        self.tracker_model = tracker_model
        self.tracker_weight = tracker_weight

        self.detector, self.short_size, self.max_size = self.initDetector()
        self.tracker = self.initTracker(
            tracker_weight=self.tracker_weight
        )

        # 是否展示数据
        self.showRes = showRes

    def initTrackInfo(self, iframe_no):
        return [{
            'frameNo': iframe_no,
            'trackid': -1,
            'boxesX1': -1,
            'boxesY1': -1,
            'boxesX2': -1,
            'boxesY2': -1,
            'conf': 0,
            'cat': 1,
            'iscrowd': 0,
        }]

    def save_track_info(self, frameNo, bbox, identities=None):
        cur_frame_track_info = []
        for i, box in enumerate(bbox):
            id = int(identities[i]) if identities is not None else 0
            x1, y1, x2, y2 = [int(i) for i in box]

            cur_frame_track_info.append({
                'frameNo': frameNo,
                'trackid': id,
                'boxesX1': x1,
                'boxesY1': y1,
                'boxesX2': x2,
                'boxesY2': y2,
                'conf': 0,
                'cat': 1,
                'iscrowd': 0,
            })
        return cur_frame_track_info

    def initDetector(self):
        '''
        自定义目标检测器
        :return:
        '''
        model_cfg = self.detector_model_config
        model = self.detector_model(model_cfg())

        state_dict = mge.load(self.detector_weight)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        print("load model file:", self.detector_weight)
        model.eval()

        detector = DetEvaluator(model)
        short_size = model.cfg.test_image_short_size
        max_size = model.cfg.test_image_max_size
        return detector, short_size, max_size

    def initTracker(self, tracker_weight):
        '''
        自定义目标跟踪器
        :return:
        '''
        max_age = tracker_weight['max_age']
        min_hits = tracker_weight['min_hits']
        iou_threshold = tracker_weight['iou_threshold']

        mot_sort = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        return mot_sort



    def multipleFishTracking(self, img_filenames, seq_name):
        # 记录整段视频轨迹信息的数组
        all_track_info = []
        for idx, img_filename in enumerate(img_filenames):
            # idx = int(img_filename.split("/")[-1].split(".")[0])
            print(img_filename)
            img = cv2.imread(img_filename)

            image, im_info = DetEvaluator.process_inputs(
                img.copy(),
                self.short_size,
                self.max_size,
            )
            pred_res = self.detector.predict(
                image=mge.tensor(image),
                im_info=mge.tensor(im_info)
            )
            if pred_res.shape[0] != 0:
                boxes = pred_res[..., :4]
                scores = pred_res[..., 4]
                # 按照每个bbox的框从大到小排序
                sorted_ind = np.argsort(-scores)
                boxes = boxes[sorted_ind]
                # 选出得分最高的fish_num个检测框
                boxes = boxes[:self.fish_num, :]

                outputs = self.tracker.update(boxes)
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                # frame的编号从1 开始，idx需+1
                cur_track_infos = self.save_track_info(idx+1, bbox_xyxy, identities)
            else:
                cur_track_infos = self.initTrackInfo(idx+1)

            if self.showRes:
                res_img = img.copy()
                for i in cur_track_infos:
                    x1, y1, x2, y2 = i['boxesX1'], i['boxesY1'], i['boxesX2'], i['boxesY2']
                    track_id = i['trackid']
                    # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
                    # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
                    res_img = cv2.rectangle(
                        res_img, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 255), 2
                    )
                    res_img = cv2.putText(
                        res_img, str(track_id),
                        (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2
                    )
                cv2.imwrite('workspace/res/track_{}_res_{}.png'.format(seq_name, str(idx)), res_img)
#                 plt.imshow(res_img)
#                 plt.show()

            all_track_info.extend(cur_track_infos)
        return all_track_info


######################################################


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_list_path",
        default='/home/huangjinze/code/data/MFTChallenge/coco/preliminary_annotations/test.json',
        type=str, help="ground truth filepath"
    )
    parser.add_argument(
        '--result_save_path',
        # default='/home/megstudio/workspace/data/preliminary/test',
        default='./preliminary_submit',
        type=str, help="ground truth filepath"
    )

    return parser

if __name__ == "__main__":
    '''
    RESULT_PATH: 参赛者上传的存储结果文件夹
    '''
    import pandas as pd
    import shutil

    parser = make_parser()
    args = parser.parse_args()

    detector_model_config = RetinaNetConfig
    detector_model = RetinaNet

    detector_weight = '../log_of_retinanet_res50_coco_3x_800size/epoch_15.pkl'
    tracker_model = 'SORT'
    tracker_weight = {
        'max_age': 3,
        'min_hits': 2,
        'iou_threshold': 0.25,
    }
    FishTracker = FishTracking(
        detector_model, detector_model_config, detector_weight,
        tracker_model, tracker_weight
    )

    RESULT_PATH = args.result_save_path

    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.mkdir(RESULT_PATH)

    COCO_PATH = args.test_list_path

    with open(COCO_PATH, 'r', encoding='utf8')as fp:
        valdata_list = json.load(fp)

    for idata in valdata_list['videos']:
        print("processing sequence:{0}".format(idata['file_name']))
        # 不抽帧时跟踪结果的存储路径
        track_result_path = os.path.join(
            RESULT_PATH, idata['file_name'] + '_track_s1_test_no1.txt'
        )
        image_file_list = idata['image_list']
        all_track_info = FishTracker.multipleFishTracking(image_file_list, idata['file_name'])
        if len(all_track_info) > 0:
            df = pd.DataFrame(all_track_info)
            df.to_csv(track_result_path, index=False, header=False)

        # 抽5帧时跟踪结果的存储路径
        track5_result_path = os.path.join(
            RESULT_PATH, idata['file_name'] + '_track_s5_test_no1.txt'
        )
        image_5file_list = idata['image_5list']
        all_track_info = FishTracker.multipleFishTracking(image_5file_list, idata['file_name'])
        if len(all_track_info) > 0:
            df = pd.DataFrame(all_track_info)
            df.to_csv(track5_result_path, index=False, header=False)




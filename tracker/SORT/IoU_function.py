# -*- coding: utf-8 -*-
#
# This is the python code for calculating bbox IoU,
# By running the script, we can get the IoU score between pred / gt bboxes
#
# Author: hzhumeng01 2018-10-19
# copyright @ netease, AI group

from __future__ import print_function, absolute_import
import numpy as np


def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(pred_bbox[2] - pred_bbox[0]) > 0) and
    #         (abs(pred_bbox[3] - pred_bbox[1]) > 0))
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    # -----0---- get coordinates of inters
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) +
           (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni

    return overlaps


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


def giou_batch(bb_test, bb_gt):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # ===========cal IOU=============#
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union_area = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    ious = wh / union_area
    # ===========cal enclose area for GIOU=============#
    enclose_left_up_xx1 = np.minimum(bb_test[..., 0], bb_gt[..., 0])
    enclose_left_up_yy1 = np.minimum(bb_test[..., 1], bb_gt[..., 1])
    enclose_right_down_xx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
    enclose_right_down_yy2 = np.maximum(bb_test[..., 3], bb_gt[..., 3])
    enclose_w = np.maximum(enclose_right_down_xx2 - enclose_left_up_xx1, 0.0)
    enclose_h = np.maximum(enclose_right_down_yy2 - enclose_left_up_yy1, 0.0)
    enclose_area = enclose_w * enclose_h

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area

    return gious


def diou_batch(bb_test, bb_gt):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # ===========cal IOU=============#
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union_area = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    ious = wh / union_area

    # cal outer boxes
    enclose_left_up_xx1 = np.minimum(bb_test[..., 0], bb_gt[..., 0])
    enclose_left_up_yy1 = np.minimum(bb_test[..., 1], bb_gt[..., 1])
    enclose_right_down_xx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
    enclose_right_down_yy2 = np.maximum(bb_test[..., 3], bb_gt[..., 3])
    enclose_w = np.maximum(enclose_right_down_xx2 - enclose_left_up_xx1, 0.0)
    enclose_h = np.maximum(enclose_right_down_yy2 - enclose_left_up_yy1, 0.0)

    outer_diagonal_line = np.square(enclose_w) + np.square(enclose_h)

    # cal center distance
    bb_test_center_x = (bb_test[..., 0] + bb_test[..., 2]) * 0.5
    bb_test_center_y = (bb_test[..., 1] + bb_test[..., 2]) * 0.5
    bb_gt_center_x = (bb_gt[..., 0] + bb_gt[..., 2]) * 0.5
    bb_gt_center_y = (bb_gt[..., 1] + bb_gt[..., 2]) * 0.5
    center_dis = np.square(bb_test_center_x - bb_gt_center_x) + \
                 np.square(bb_test_center_y - bb_gt_center_y)

    # cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious


def ciou_batch(bb_test, bb_gt):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    # ===========cal IOU=============#
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union_area = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    ious = wh / union_area

    # cal outer boxes
    enclose_left_up_xx1 = np.minimum(bb_test[..., 0], bb_gt[..., 0])
    enclose_left_up_yy1 = np.minimum(bb_test[..., 1], bb_gt[..., 1])
    enclose_right_down_xx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
    enclose_right_down_yy2 = np.maximum(bb_test[..., 3], bb_gt[..., 3])
    enclose_w = np.maximum(enclose_right_down_xx2 - enclose_left_up_xx1, 0.0)
    enclose_h = np.maximum(enclose_right_down_yy2 - enclose_left_up_yy1, 0.0)

    outer_diagonal_line = np.square(enclose_w) + np.square(enclose_h)

    # cal center distance
    bb_test_center_x = (bb_test[..., 0] + bb_test[..., 2]) * 0.5
    bb_test_center_y = (bb_test[..., 1] + bb_test[..., 2]) * 0.5
    bb_gt_center_x = (bb_gt[..., 0] + bb_gt[..., 2]) * 0.5
    bb_gt_center_y = (bb_gt[..., 1] + bb_gt[..., 2]) * 0.5
    center_dis = np.square(bb_test_center_x - bb_gt_center_x) + \
                 np.square(bb_test_center_y - bb_gt_center_y)

    # cal penalty term
    # cal width,height

    bb_gt_w = np.maximum(bb_gt[..., 2] - bb_gt[..., 0], 0.0)
    bb_gt_h = np.maximum(bb_gt[..., 3] - bb_gt[..., 1], 0.0)
    bb_test_w = np.maximum(bb_test[..., 2] - bb_test[..., 0], 0.0)
    bb_test_h = np.maximum(bb_test[..., 3] - bb_test[..., 1], 0.0)

    v = (4.0 / np.square(np.pi)) * np.square((
            np.arctan((bb_gt_w / bb_gt_h)) -
            np.arctan((bb_test_w / bb_test_h))))
    alpha = v / (1 - ious + v)
    # cal cious
    cious = ious - (center_dis / outer_diagonal_line + alpha * v)

    return cious


def get_max_IoU(pred_bboxes, gt_bbox):
    """
    given 1 gt bbox, >1 pred bboxes, return max iou score for the given gt bbox and pred_bboxes
    :param pred_bbox: predict bboxes coordinates, we need to find the max iou score with gt bbox for these pred bboxes
    :param gt_bbox: ground truth bbox coordinate
    :return: max iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    if pred_bboxes.shape[0] > 0:
        # -----0---- get coordinates of inters, but with multiple predict bboxes
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # -----1----- intersection
        inters = iw * ih

        # -----2----- union, uni = S1 + S2 - inters
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)

        # -----3----- iou, get max score and max iou index
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    return overlaps, ovmax, jmax

def use_iou(IOU_strategy, bb_test, bb_gt):
    if IOU_strategy == 'iou':
        return iou_batch(bb_test[:, :4], bb_gt)
    elif IOU_strategy == 'giou':
        return giou_batch(bb_test[:, :4], bb_gt)
    elif IOU_strategy == 'diou':
        return diou_batch(bb_test[:, :4], bb_gt)
    elif IOU_strategy == 'ciou':
        return ciou_batch(bb_test[:, :4], bb_gt)

if __name__ == "__main__":
    # test1
    # pred_bbox = np.array([50, 50, 90, 100])  # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    # gt_bbox = np.array([70, 80, 120, 150])
    # print(get_IoU(pred_bbox, gt_bbox))
    pred_bbox = np.loadtxt('testData/trackers.csv', delimiter=",", dtype="float")
    gt_bbox = np.loadtxt('testData/detections.csv', delimiter=",", dtype="float")

    print(iou_batch(pred_bbox[:, :4], gt_bbox))
    # print(giou_batch(pred_bbox[:, :4], gt_bbox))
    # print(diou_batch(pred_bbox[:, :4], gt_bbox))
    print(ciou_batch(pred_bbox[:, :4], gt_bbox))

    # # test2
    # pred_bboxes = np.array([[15, 18, 47, 60],
    #                         [50, 50, 90, 100],
    #                         [70, 80, 120, 145],
    #                         [130, 160, 250, 280],
    #                         [25.6, 66.1, 113.3, 147.8]])
    # gt_bbox = np.array([70, 80, 120, 150])
    # print(get_max_IoU(pred_bboxes, gt_bbox))

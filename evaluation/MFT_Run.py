import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import shutil
import cv2

race_path = 'preliminary' # 'preliminary' or 'intermediary'
DATA_PATH = os.path.join('workspace/data', race_path)

class FishTracking(object):
    def __init__(self,
                 detector_model,  detector_weight,
                 tracker_model, tracker_weight,
                 fish_num=20, showRes=True
                 ):
        self.fish_num = fish_num

        self.detector_model = detector_model
        self.detector_weight = detector_weight

        self.tracker_model = tracker_model
        self.tracker_weight = tracker_weight


        self.detector = self.initDetector()
        self.tracker = self.initTracker()

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

    def save_track_info(self, frameNo, bbox, identities=None, score=0):
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
                'conf': score,
                'cat': 1,
                'iscrowd': 0,
            })
        return cur_frame_track_info

    def initDetector(self, detector_model, detector_weight):
        '''
        自定义目标检测器
        :return:
        '''


    def initTracker(self, tracker_model, tracker_weight):
        '''
        自定义目标跟踪器
        :return:
        '''



    def multipleFishTracking(self, img_filenames, seq_name):
        # 记录整段视频轨迹信息的数组
        all_track_info = []
        for idx, img_filename in enumerate(img_filenames):
            img = cv2.imread(img_filename)

            pred_res = self.detector()

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

            all_track_info.extend(cur_track_infos)
        return all_track_info


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weight_path",
        default='/home/megstudio/workspace/log_of_retinanet_res50_coco_3x_800size/epoch_1.pkl',
        type=str, help="track_result filepath"
    )
    parser.add_argument(
        "--test_seqs_path",
        default='/home/megstudio/workspace/data/preliminary/test',
        type=str, help="ground truth filepath"
    )
    parser.add_argument(
        'workspace/TrackResult',
        default='/home/megstudio/workspace/data/preliminary/test',
        type=str, help="ground truth filepath"
    )

    return parser

if __name__ == "__main__":
    '''
    RESULT_PATH: 参赛者上传的存储结果文件夹
    '''
    parser = make_parser()
    args = parser.parse_args()
    RESULT_PATH = args.RESULT_PATH

    detector_weight = 'path/to/detector/weight'
    tracker_weight = 'path/to/tracker/weight'

    FishTracker = FishTracking(
        detector_model, detector_weight,
        tracker_model, tracker_weight
    )


    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.mkdir(RESULT_PATH)

    COCO_PATH = os.path.join(
        DATA_PATH, '../coco',
        race_path + '_annotations', 'test.json'
    )

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



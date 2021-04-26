from tools.utils import DetEvaluator, import_from_file
from tracker.SORT.sort import Sort
import megengine as mge
import os
import cv2
import numpy as np
from tools.utils import save_track_info
import pandas as pd
import json

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data/')

def getValImgList():
    COCO_PATH = os.path.join(
        DATA_PATH, 'coco',
        'annotations', 'val_half.json'
    )
    with open(COCO_PATH, 'r', encoding='utf8')as fp:
        val_data = json.load(fp)
    return val_data['videos']


class FishTracking(object):
    def __init__(self,
                 detector_model, detector_weight,
                 tracker_model, tracker_weight,
                 fish_num=20, showRes=False
                 ):
        self.fish_num = fish_num

        self.detector_model = detector_model
        self.detector_weight = detector_weight

        self.tracker_model = tracker_model
        self.tracker_weight = tracker_weight

        self.detector, self.short_size, self.max_size = self.initDetector()
        self.tracker = self.initTracker(
            tracker_weight=self.tracker_weight
        )
        # 记录整段视频轨迹信息的数组
        self.all_track_info = []
        # 是否展示数据
        self.showRes = showRes

    def initTrackInfo(self, iframe_no):
        return [{
            'frame_id': iframe_no,
            'track_id': -1,
            'boxesX1': -1,
            'boxesY1': -1,
            'boxesX2': -1,
            'boxesY2': -1,
            'centerX': -1,
            'centerY': -1,
        }]

    def initDetector(self):
        '''
        自定义目标检测器
        :return:
        '''
        current_network = import_from_file(self.detector_model)
        cfg = current_network.Cfg()
        cfg.backbone_pretrained = False
        model = current_network.Net(cfg)
        model.eval()

        state_dict = mge.load(self.detector_weight)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

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



    def multipleFishTracking(self, img_filenames):
        for idx, img_filename in enumerate(img_filenames):
            print("processing file:{0}".format(img_filename))
            # idx = int(img_filename.split("/")[-1].split(".")[0])
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
                cur_track_infos = save_track_info(idx, bbox_xyxy, identities)
            else:
                cur_track_infos = self.initTrackInfo(idx)

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
                    # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
                cv2.imwrite('track_res_{}.png'.format(str(idx)), res_img)

            self.all_track_info.extend(cur_track_infos)
        return self.all_track_info


if __name__ == '__main__':
    detector_model = 'detector/configs/retinanet_res50_coco_3x_800size.py'
    detector_weight = 'detector/log/log-of-retinanet_res50_coco_3x_800size/epoch_10.pkl'
    tracker_model = 'SORT'
    tracker_weight = {
        'max_age': 3,
        'min_hits': 2,
        'iou_threshold': 0.25,
    }
    FishTracker = FishTracking(
        detector_model, detector_weight,
        tracker_model, tracker_weight
    )
    valdata_list = getValImgList()

    for idata in valdata_list:
        # 不抽帧时跟踪结果的存储路径
        track_result_path = os.path.join(
            DATA_PATH, 'train',
            idata['file_name'], 'gt', 'track_res.txt'
        )
        image_file_list = idata['image_list']
        all_track_info = FishTracker.multipleFishTracking(image_file_list)
        if len(all_track_info) > 0:
            df = pd.DataFrame(all_track_info)
            df.to_csv(track_result_path, index=False)

        # 抽5帧时跟踪结果的存储路径
        track5_result_path = os.path.join(
            DATA_PATH, 'train',
            idata['file_name'], 'gt', 'track_5_res.txt'
        )
        image_5file_list = idata['image_5list']
        all_track_info = FishTracker.multipleFishTracking(image_5file_list)
        if len(all_track_info) > 0:
            df = pd.DataFrame(all_track_info)
            df.to_csv(track5_result_path, index=False, header=0)




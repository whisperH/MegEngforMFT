import cv2
import os
import json
import numpy as np
import pandas as pd

split_flag = 'val'

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, '../data/')

COCO_PATH = os.path.join(
    DATA_PATH, 'coco',
    'annotations', '{}_half.json'.format(split_flag)
)


class showDemoData(object):
    def __init__(self,
                 show_image_nums=10,
                 image_freq=1,
                 start_img_id=1
                 ):
        with open(COCO_PATH, 'r', encoding='utf8')as fp:
            self.data = json.load(fp)

        image_nums = len(self.data['images'])
        assert show_image_nums > 0, 'show_image_nums must in (0, {})'.format(image_nums)
        assert image_freq == 1 or image_freq == 5, 'image_freq must be 1 or 5'
        assert start_img_id > 0 and start_img_id + image_freq < image_nums, 'image_freq must be 1 or 5'

        self.show_image_nums = show_image_nums
        self.image_freq = image_freq
        self.start_img_id = start_img_id
        self.end_img_id = start_img_id + image_freq * show_image_nums

    def getImgInfo(self, Img_ids: list):
        assert len(Img_ids) > 1, 'list: Img_ids must contain 2 image id'
        Img_infos = {}

        for iImg in self.data['images']:
            if iImg['id'] in Img_ids:
                Img_infos[iImg['id']] = iImg
            else:
                continue
        return sorted(Img_infos.items(), key=lambda d: d[0])

    def getCOCOAnnoInfo(self, Img_ids: list):
        assert len(Img_ids) > 1, 'list: Img_ids must contain 2 image id'
        Anno_infos = {}

        for iAnno in self.data['annotations']:
            if iAnno['image_id'] in Img_ids:
                if iAnno['image_id'] not in Anno_infos:
                    Anno_infos[iAnno['image_id']] = []
                Anno_infos[iAnno['image_id']].append(iAnno)
            else:
                continue
        return sorted(Anno_infos.items(), key=lambda d: d[0])



    def drawCOCOAnnoInfoImg(self, Img_info, Anno_info, Img_id):
        Img_path = Img_info['file_name']
        image = cv2.imread(Img_path)
        for i in range(len(Anno_info)):
            annotation = Anno_info[i]
            bbox = annotation['bbox']  # (x1, y1, w, h)
            x, y, w, h = bbox

            # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
            # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
            image = cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)),
                (0, 255, 255), 2
            )
            image = cv2.putText(
                image, str(annotation['track_id']),
                (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2
            )
            # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
        cv2.imwrite('demo{}.png'.format(str(Img_id)), image)
        # cv2.waitKey(5000)

    def run(self):
        Img_ids = [_ for _ in range(self.start_img_id, self.end_img_id, self.image_freq)]
        Img_infos = self.getImgInfo(Img_ids)
        Anno_infos = self.getCOCOAnnoInfo(Img_ids)

        for idx, image_id in enumerate(Img_ids):
            self.drawCOCOAnnoInfoImg(Img_infos[idx][1], Anno_infos[idx][1], image_id)
            # testUnit(Img_infos[idx][1], Anno_infos[idx][1])
            print(Anno_infos[idx][1])
            print("=====================")

# class showTrackingRes(object):
#     def __init__(self, image_list, data_type='5_val'):
#         '''
#
#         :param image_list:
#         :param data_type:
#             val: 代表人工标注数据
#             5_val:代表人工标注抽取5帧的数据
#         '''
#         self.image_list = image_list
#         # 读取MOT格式的数据，标注或者是跟踪
#         MOT_Anno_path = os.path.split(
#             image_list[0].replace("img1", "gt")
#         )[0]
#         self.MOT_Anno_filepath = os.path.join(
#             MOT_Anno_path,
#             "gt_{}_half.txt".format(data_type)
#         )
#         MOT_Anno = np.loadtxt(
#             self.MOT_Anno_filepath,
#             dtype=np.float32,
#             delimiter=','
#         )
#         for idx, iimage_file in enumerate(self.image_list):
#             img = cv2.imread(iimage_file)
#
#             imot_data = MOT_Anno[MOT_Anno[:, 0] == (idx+1)]
#
#             for i in range(len(imot_data)):
#                 annotation = imot_data[i, :]
#                 x1, y1, x2, y2 = annotation[2:6]
#                 # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
#                 # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
#                 img = cv2.rectangle(
#                     img, (int(x1), int(y1)), (int(x2), int(y2)),
#                     (0, 255, 255), 2
#                 )
#                 img = cv2.putText(
#                     img, str(annotation[1]),
#                     (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (255, 0, 255), 2
#                 )
#                 # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
#             cv2.imwrite('demo{}.png'.format(str(idx)), img)


if __name__ == '__main__':
    showDemoData(
        show_image_nums=4,
        image_freq=5,
        start_img_id=451
    ).run()


import os
import numpy as np
import json
import cv2

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, '../data/')
# SPLITS = ['train_half', 'val_half', 'train', 'test']
SPLITS = ['train_half', 'val_half']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True


def gen_COCO_Anno(data_path, out_path):
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'fish'}],
           'videos': []}
    # seqs: [track1, track2, ..., files under the folder "train"]
    seqs = os.listdir(data_path)
    # image's counter
    image_cnt = 0
    # annotation's counter
    ann_cnt = 0
    # sub-dataset's counter
    video_cnt = 0
    # traverse each sub_dataset
    for seq in sorted(seqs):
        video_cnt += 1

        # locate track"i"/
        seq_path = '{}/{}/'.format(data_path, seq)
        img_path = seq_path + 'img1/'
        ann_path = seq_path + 'gt/gt.txt'
        images = os.listdir(img_path)

        num_images = len([image for image in images if 'PNG' in image])
        if HALF_VIDEO and ('half' in split):
            image_range = [0, num_images // 2] if 'train' in split else \
                [num_images // 2 + 1, num_images - 1]
        else:
            image_range = [0, num_images - 1]

        # 用于存储单帧验证数据的文件列表
        image_filelist = []
        # 用于存储抽5帧验证数据的文件列表
        image_5filelist = []

        for i in range(num_images):
            if (i < image_range[0] or i > image_range[1]):
                continue
            img_filename = img_path + '{:06d}.PNG'.format(i + 1)
            img = cv2.imread(img_filename)
            image_info = {
                'file_name': img_filename,
                'id': image_cnt + i + 1,
                'frame_id': i + 1 - image_range[0],
                'prev_image_id': image_cnt + i if i > 0 else -1,
                'prev_5image_id': image_cnt + i - 4 if i - 5 > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                'next_5image_id': image_cnt + i + 6 if i < num_images - 5 else -1,
                'video_id': video_cnt,
                'width': img.shape[1],
                'height': img.shape[0],
            }
            image_filelist.append(img_filename)
            if i % 5 == 0:
                image_5filelist.append(img_filename)
            out['images'].append(image_info)

        out['videos'].append({
            'id': video_cnt,
            'file_name': seq,
            'image_list': image_filelist,
            'image_5list': image_5filelist,
        })
        print('{}: {} images'.format(seq, num_images))

        if split != 'test':
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            anns = anns[np.argsort(anns[:, 0])]

            # 生成划分数据后的MOT格式标注文件：
            anns_out = np.array([anns[i] for i in range(anns.shape[0]) if \
                                 int(anns[i][0]) - 1 >= image_range[0] and \
                                 int(anns[i][0]) - 1 <= image_range[1]], np.float32)
            anns_out[:, 0] -= image_range[0]
            # 生成的单帧文件
            gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
            fout = open(gt_out, 'w')
            for o in anns_out:
                fout.write(
                    '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                        int(o[6]), int(o[7]), o[8]))
            fout.close()

            # 生成的抽帧后的文件
            gt_out = seq_path + '/gt/gt_5_{}.txt'.format(split)
            fout = open(gt_out, 'w')
            for idx, o in enumerate(anns_out):
                if idx % 5 == 0:
                    fout.write(
                        '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                            int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                            int(o[6]), int(o[7]), o[8]))
            fout.close()

            # 生成 COCO格式
            print(' {} ann images'.format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                    continue
                track_id = int(anns[i][1])
                ann_cnt += 1

                category_id = 1
                area = (anns[i][4] - anns[i][2]) * (anns[i][5] - anns[i][3])

                ann = {
                    'id': ann_cnt,
                    'category_id': category_id,
                    'image_id': image_cnt + frame_id,
                    'track_id': track_id,
                    'bbox': anns[i][2:6].tolist(),
                    'conf': float(anns[i][6]),
                    'area': area.item() / 2.0,
                    'iscrowd': int(anns[i][8])
                }
                out['annotations'].append(ann)
        image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
        split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
    for split in SPLITS:
        # data_path: data/train
        data_path = DATA_PATH + (split if not HALF_VIDEO else 'train')
        OUT_PATH = os.path.join(DATA_PATH, 'coco', 'annotations')

        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)

        out_path = OUT_PATH + '/{}.json'.format(split)
        gen_COCO_Anno(data_path, out_path)

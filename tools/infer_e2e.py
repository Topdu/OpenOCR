from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
import numpy as np
import copy
import time
import cv2
import json
from PIL import Image
import torch
from tools.utils.utility import get_image_file_list, check_and_read
from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop, draw_ocr_box_txt


def set_device(device):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def check_and_download_font(font_path):
    if not os.path.exists(font_path):
        print(f"Downloading '{font_path}' ...")
        try:
            import urllib.request
            font_url = 'https://shuiche-shop.oss-cn-chengdu.aliyuncs.com/fonts/simfang.ttf'
            urllib.request.urlretrieve(font_url, font_path)
            print(f'Downloading font success: {font_path}')
        except Exception as e:
            print(f'Downloading font error: {e}')


class OpenOCR(object):

    def __init__(self, mode='mobile', drop_score=0.5, det_box_type='quad'):
        """
        初始化函数，用于初始化OCR引擎的相关配置和组件。

        Args:
            mode (str, optional): 运行模式，可选值为'mobile'或'server'。默认为'mobile'。
            drop_score (float, optional): 检测框的置信度阈值，低于该阈值的检测框将被丢弃。默认为0.5。
            det_box_type (str, optional): 检测框的类型，可选值为'quad' and 'poly'。默认为'quad'。

        Returns:
            无返回值。

        """
        cfg_det = Config(
            './configs/det/dbnet/repvit_db.yml').cfg  # mobile model
        if mode == 'server':
            cfg_rec = Config(
                './configs/det/svtrv2/svtrv2_ch.yml').cfg  # server model
        else:
            cfg_rec = Config(
                './configs/rec/svtrv2/repsvtr_ch.yml').cfg  # mobile model
        self.text_detector = OpenDetector(cfg_det)
        self.text_recognizer = OpenRecognizer(cfg_rec)
        self.det_box_type = det_box_type
        self.drop_score = drop_score

        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f'mg_crop_{bno+self.crop_image_res_index}.jpg'),
                img_crop_list[bno],
            )
        self.crop_image_res_index += bbox_num

    def infer_single_image(self,
                           img_numpy,
                           ori_img,
                           crop_infer=False,
                           rec_batch_num=6):
        start = time.time()
        if crop_infer:
            dt_boxes = self.text_detector.crop_infer(
                img_numpy=img_numpy)[0]['boxes']
        else:
            dt_boxes = self.text_detector(img_numpy=img_numpy)[0]['boxes']
        # print(dt_boxes)
        det_time_cost = time.time() - start

        if dt_boxes is None:
            return None

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = np.array(copy.deepcopy(dt_boxes[bno])).astype(np.float32)
            if self.det_box_type == 'quad':
                img_crop = get_rotate_crop_image(ori_img, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_img, tmp_box)
            img_crop_list.append(img_crop)

        start = time.time()
        rec_res = self.text_recognizer(img_numpy_list=img_crop_list,
                                       batch_num=rec_batch_num)
        rec_time_cost = time.time() - start

        filter_boxes, filter_rec_res = [], []
        rec_time_cost_sig = 0.0
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result['text'], rec_result['score']
            rec_time_cost_sig += rec_result['elapse']
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append([text, score])

        avg_rec_time_cost = rec_time_cost_sig / len(dt_boxes) if len(
            dt_boxes) > 0 else 0.0

        return filter_boxes, filter_rec_res, {
            'time_cost': det_time_cost + rec_time_cost,
            'detection_time': det_time_cost,
            'recognition_time': rec_time_cost,
            'avg_rec_time_cost': avg_rec_time_cost
        }

    def __call__(self,
                 img_path=None,
                 img_numpy=None,
                 rec_batch_num=6,
                 crop_infer=False):
        """
        对输入的图像进行文本识别，并返回识别结果和时间信息。

        Args:
            img_path (str, optional): 输入图像的路径。默认为None。
            img_numpy (numpy.ndarray, list[numpy.ndarray], optional): 输入的图像数据，以numpy数组的形式表示。默认为None。
            crop_infer (bool, optional): 是否进行裁剪推理。默认为False。
            rec_batch_num (int, optional): 识别模型批量推理时的批大小。默认为6。

        Returns:
            tuple: 返回一个元组，包含两个元素。
                - list: 包含识别结果的列表，每个元素是一个字典，包含文本转录、文本框坐标点和得分。
                - dict: 包含时间信息的字典。

        """

        if img_numpy is None:
            img, flag_gif, flag_pdf = check_and_read(img_path)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(img_path)
            if not flag_pdf:
                if img is None:
                    return None
                imgs = [img]
            else:
                imgs = img
        else:
            if isinstance(img_numpy, list):
                imgs = img_numpy
            else:
                imgs = [img_numpy]
        # for img_numpy in imgs:
        results = []
        for index, img_numpy in enumerate(imgs):
            ori_img = img_numpy.copy()
            dt_boxes, rec_res, time_dict = self.infer_single_image(
                img_numpy=img_numpy,
                ori_img=ori_img,
                crop_infer=crop_infer,
                rec_batch_num=rec_batch_num)

            res = [{
                'transcription': rec_res[i][0],
                'points': np.array(dt_boxes[i]).tolist(),
                'score': rec_res[i][1],
            } for i in range(len(dt_boxes))]
            results.append(res)
        return results, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main():
    img_path = './test_img/'
    image_file_list = get_image_file_list(img_path)
    drop_score = 0.5
    text_sys = OpenOCR(drop_score=drop_score,
                       det_box_type='quad')  # det_box_type: 'quad' or 'poly'
    is_visualize = False
    if is_visualize:
        font_path = './simfang.ttf'
        check_and_download_font(font_path)
    draw_img_save_dir = img_path + 'e2e_results/' if img_path[
        -1] != '/' else img_path[:-1] + 'e2e_results/'
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                return None
            imgs = [img]
        else:
            imgs = img

        res_list, time_dict = text_sys(img_numpy=imgs)
        print(time_dict)

        for index, res in enumerate(res_list):

            if len(res_list) > 1:
                save_pred = (os.path.basename(image_file) + '_' + str(index) +
                             '\t' + json.dumps(res, ensure_ascii=False) + '\n')
            else:
                save_pred = (os.path.basename(image_file) + '\t' +
                             json.dumps(res, ensure_ascii=False) + '\n')
            save_results.append(save_pred)
            dt_boxes = [res[i]['points'] for i in range(len(res))]
            rec_res = [res[i]['transcription'] for i in range(len(res))]
            rec_score = [res[i]['score'] for i in range(len(res))]
            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i] for i in range(len(rec_res))]
                scores = [rec_score[i] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path,
                )
                if flag_gif:
                    save_file = image_file[:-3] + 'png'
                elif flag_pdf:
                    save_file = image_file.replace('.pdf',
                                                   '_' + str(index) + '.png')
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir,
                                 os.path.basename(save_file)),
                    draw_img[:, :, ::-1],
                )

    with open(os.path.join(draw_img_save_dir, 'system_results.txt'),
              'w',
              encoding='utf-8') as f:
        f.writelines(save_results)


if __name__ == '__main__':
    main()

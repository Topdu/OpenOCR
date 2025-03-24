from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
import argparse
import numpy as np
import copy
import time
import cv2
import json
from PIL import Image
from tools.utils.utility import get_image_file_list, check_and_read
from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine.config import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop, draw_ocr_box_txt
from tools.utils.logging import get_logger

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH_DET = str(root_dir / '../configs/det/dbnet/repvit_db.yml')
DEFAULT_CFG_PATH_REC_SERVER = str(root_dir /
                                  '../configs/rec/svtrv2/svtrv2_ch.yml')
DEFAULT_CFG_PATH_REC = str(root_dir / '../configs/rec/svtrv2/repsvtr_ch.yml')

logger = get_logger()


def check_and_download_font(font_path):
    if not os.path.exists(font_path):
        cache_dir = Path.home() / '.cache' / 'openocr'
        font_path = str(cache_dir / font_path)
        if os.path.exists(font_path):
            return font_path
        logger.info(f"Downloading '{font_path}' ...")
        try:
            import urllib.request
            font_url = 'https://shuiche-shop.oss-cn-chengdu.aliyuncs.com/fonts/simfang.ttf'
            urllib.request.urlretrieve(font_url, font_path)
            logger.info(f'Downloading font success: {font_path}')
        except Exception as e:
            logger.info(f'Downloading font error: {e}')
    return font_path


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


class OpenOCR(object):

    def __init__(self,
                 mode='mobile',
                 backend='torch',
                 onnx_det_model_path=None,
                 onnx_rec_model_path=None,
                 drop_score=0.5,
                 det_box_type='quad',
                 device='gpu'):
        """
        初始化函数，用于初始化OCR引擎的相关配置和组件。

        Args:
            mode (str, optional): 运行模式，可选值为'mobile'或'server'。默认为'mobile'。
            drop_score (float, optional): 检测框的置信度阈值，低于该阈值的检测框将被丢弃。默认为0.5。
            det_box_type (str, optional): 检测框的类型，可选值为'quad' and 'poly'。默认为'quad'。

        Returns:
            无返回值。

        """
        cfg_det = Config(DEFAULT_CFG_PATH_DET).cfg  # mobile model
        cfg_det['Global']['device'] = device
        if mode == 'server':
            cfg_rec = Config(DEFAULT_CFG_PATH_REC_SERVER).cfg  # server model
        else:
            cfg_rec = Config(DEFAULT_CFG_PATH_REC).cfg  # mobile model

        cfg_rec['Global']['device'] = device

        self.text_detector = OpenDetector(cfg_det,
                                          backend=backend,
                                          onnx_model_path=onnx_det_model_path)
        self.text_recognizer = OpenRecognizer(
            cfg_rec, backend=backend, onnx_model_path=onnx_rec_model_path)
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
                           rec_batch_num=6,
                           return_mask=False,
                           **kwargs):
        start = time.time()
        if crop_infer:
            dt_boxes = self.text_detector.crop_infer(
                img_numpy=img_numpy)[0]['boxes']
        else:
            det_res = self.text_detector(img_numpy=img_numpy,
                                         return_mask=return_mask,
                                         **kwargs)[0]
            dt_boxes = det_res['boxes']
        # logger.info(dt_boxes)
        det_time_cost = time.time() - start

        if dt_boxes is None:
            return None, None, None

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
        if return_mask:
            return filter_boxes, filter_rec_res, {
                'time_cost': det_time_cost + rec_time_cost,
                'detection_time': det_time_cost,
                'recognition_time': rec_time_cost,
                'avg_rec_time_cost': avg_rec_time_cost
            }, det_res['mask']

        return filter_boxes, filter_rec_res, {
            'time_cost': det_time_cost + rec_time_cost,
            'detection_time': det_time_cost,
            'recognition_time': rec_time_cost,
            'avg_rec_time_cost': avg_rec_time_cost
        }

    def __call__(self,
                 img_path=None,
                 save_dir='e2e_results/',
                 is_visualize=False,
                 img_numpy=None,
                 rec_batch_num=6,
                 crop_infer=False,
                 return_mask=False,
                 **kwargs):
        """
        img_path: str, optional, default=None
            Path to the directory containing images or the image filename.
        save_dir: str, optional, default='e2e_results/'
            Directory to save prediction and visualization results. Defaults to a subfolder in img_path.
        is_visualize: bool, optional, default=False
            Visualize the results.
        img_numpy: numpy or list[numpy], optional, default=None
            numpy of an image or List of numpy arrays representing images.
        rec_batch_num: int, optional, default=6
            Batch size for text recognition.
        crop_infer: bool, optional, default=False
            Whether to use crop inference.
        """

        if img_numpy is None and img_path is None:
            raise ValueError('img_path and img_numpy cannot be both None.')
        if img_numpy is not None:
            if not isinstance(img_numpy, list):
                img_numpy = [img_numpy]
            results = []
            time_dicts = []
            for index, img in enumerate(img_numpy):
                ori_img = img.copy()
                if return_mask:
                    dt_boxes, rec_res, time_dict, mask = self.infer_single_image(
                        img_numpy=img,
                        ori_img=ori_img,
                        crop_infer=crop_infer,
                        rec_batch_num=rec_batch_num,
                        return_mask=return_mask,
                        **kwargs)
                else:
                    dt_boxes, rec_res, time_dict = self.infer_single_image(
                        img_numpy=img,
                        ori_img=ori_img,
                        crop_infer=crop_infer,
                        rec_batch_num=rec_batch_num,
                        **kwargs)
                if dt_boxes is None:
                    results.append([])
                    time_dicts.append({})
                    continue
                res = [{
                    'transcription': rec_res[i][0],
                    'points': np.array(dt_boxes[i]).tolist(),
                    'score': rec_res[i][1],
                } for i in range(len(dt_boxes))]
                results.append(res)
                time_dicts.append(time_dict)
            if return_mask:
                return results, time_dicts, mask
            return results, time_dicts

        image_file_list = get_image_file_list(img_path)
        save_results = []
        time_dicts_return = []
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
            logger.info(
                f'Processing {idx+1}/{len(image_file_list)}: {image_file}')

            res_list = []
            time_dicts = []
            for index, img_numpy in enumerate(imgs):
                ori_img = img_numpy.copy()
                dt_boxes, rec_res, time_dict = self.infer_single_image(
                    img_numpy=img_numpy,
                    ori_img=ori_img,
                    crop_infer=crop_infer,
                    rec_batch_num=rec_batch_num,
                    **kwargs)
                if dt_boxes is None:
                    res_list.append([])
                    time_dicts.append({})
                    continue
                res = [{
                    'transcription': rec_res[i][0],
                    'points': np.array(dt_boxes[i]).tolist(),
                    'score': rec_res[i][1],
                } for i in range(len(dt_boxes))]
                res_list.append(res)
                time_dicts.append(time_dict)

            for index, (res, time_dict) in enumerate(zip(res_list,
                                                         time_dicts)):

                if len(res) > 0:
                    logger.info(f'Results: {res}.')
                    logger.info(f'Time cost: {time_dict}.')
                else:
                    logger.info('No text detected.')

                if len(res_list) > 1:
                    save_pred = (os.path.basename(image_file) + '_' +
                                 str(index) + '\t' +
                                 json.dumps(res, ensure_ascii=False) + '\n')
                else:
                    if len(res) > 0:
                        save_pred = (os.path.basename(image_file) + '\t' +
                                     json.dumps(res, ensure_ascii=False) +
                                     '\n')
                    else:
                        continue
                save_results.append(save_pred)
                time_dicts_return.append(time_dict)

                if is_visualize and len(res) > 0:
                    if idx == 0:
                        font_path = './simfang.ttf'
                        font_path = check_and_download_font(font_path)
                        os.makedirs(save_dir, exist_ok=True)
                        draw_img_save_dir = os.path.join(
                            save_dir, 'vis_results/')
                        os.makedirs(draw_img_save_dir, exist_ok=True)
                        logger.info(
                            f'Visualized results will be saved to {draw_img_save_dir}.'
                        )
                    dt_boxes = [res[i]['points'] for i in range(len(res))]
                    rec_res = [
                        res[i]['transcription'] for i in range(len(res))
                    ]
                    rec_score = [res[i]['score'] for i in range(len(res))]
                    image = Image.fromarray(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i] for i in range(len(rec_res))]
                    scores = [rec_score[i] for i in range(len(rec_res))]

                    draw_img = draw_ocr_box_txt(
                        image,
                        boxes,
                        txts,
                        scores,
                        drop_score=self.drop_score,
                        font_path=font_path,
                    )
                    if flag_gif:
                        save_file = image_file[:-3] + 'png'
                    elif flag_pdf:
                        save_file = image_file.replace(
                            '.pdf', '_' + str(index) + '.png')
                    else:
                        save_file = image_file
                    cv2.imwrite(
                        os.path.join(draw_img_save_dir,
                                     os.path.basename(save_file)),
                        draw_img[:, :, ::-1],
                    )

        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'system_results.txt'),
                      'w',
                      encoding='utf-8') as f:
                f.writelines(save_results)
            logger.info(
                f"Results saved to {os.path.join(save_dir, 'system_results.txt')}."
            )
            if is_visualize:
                logger.info(
                    f'Visualized results saved to {draw_img_save_dir}.')
            return save_results, time_dicts_return
        else:
            logger.info('No text detected.')
            return None, None


def main():
    parser = argparse.ArgumentParser(description='OpenOCR system')
    parser.add_argument(
        '--img_path',
        type=str,
        help='Path to the directory containing images or the image filename.')
    parser.add_argument(
        '--mode',
        type=str,
        default='mobile',
        help="Mode of the OCR system, e.g., 'mobile' or 'server'.")
    parser.add_argument(
        '--backend',
        type=str,
        default='torch',
        help="Backend of the OCR system, e.g., 'torch' or 'onnx'.")
    parser.add_argument('--onnx_det_model_path',
                        type=str,
                        default=None,
                        help='Path to the ONNX model for text detection.')
    parser.add_argument('--onnx_rec_model_path',
                        type=str,
                        default=None,
                        help='Path to the ONNX model for text recognition.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='e2e_results/',
        help='Directory to save prediction and visualization results. \
            Defaults to ./e2e_results/.')
    parser.add_argument('--is_vis',
                        action='store_true',
                        default=False,
                        help='Visualize the results.')
    parser.add_argument('--drop_score',
                        type=float,
                        default=0.5,
                        help='Score threshold for text recognition.')
    parser.add_argument('--device',
                        type=str,
                        default='gpu',
                        help='Device to use for inference.')
    args = parser.parse_args()

    img_path = args.img_path
    mode = args.mode
    backend = args.backend
    onnx_det_model_path = args.onnx_det_model_path
    onnx_rec_model_path = args.onnx_rec_model_path
    save_dir = args.save_dir
    is_visualize = args.is_vis
    drop_score = args.drop_score
    device = args.device

    text_sys = OpenOCR(mode=mode,
                       backend=backend,
                       onnx_det_model_path=onnx_det_model_path,
                       onnx_rec_model_path=onnx_rec_model_path,
                       drop_score=drop_score,
                       det_box_type='quad',
                       device=device)  # det_box_type: 'quad' or 'poly'
    text_sys(img_path=img_path, save_dir=save_dir, is_visualize=is_visualize)


if __name__ == '__main__':
    main()

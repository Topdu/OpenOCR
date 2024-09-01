import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
import json

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from tools.engine import Config
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list


class RatioRecTVReisze(object):

    def __init__(self, cfg):
        self.max_ratio = cfg['Eval']['loader'].get('max_ratio', 12)
        self.base_shape = cfg['Eval']['dataset'].get('base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = cfg['Eval']['dataset'].get('base_h', 32)
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)
        self.ceil = cfg['Eval']['dataset'].get('ceil', False), 

    def __call__(self, data):
        img = data['image']
        imgH = self.base_h
        w, h = img.size
        if self.ceil:
            gen_ratio = int(float(w) / float(h)) + 1
        else:
            gen_ratio = max(1, round(float(w) / float(h)))
        ratio_resize = min(gen_ratio, self.max_ratio)
        imgW, imgH = self.base_shape[ratio_resize -
                                     1] if ratio_resize <= 4 else [
                                         self.base_h *
                                         ratio_resize, self.base_h
                                     ]
        resized_w = imgW
        resized_image = F.resize(img, (imgH, resized_w),
                                 interpolation=self.interpolation)
        img = self.transforms(resized_image)
        data['image'] = img
        return data


def build_rec_process(cfg):
    transforms = []
    ratio_resize_flag = True
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Resize' in op_name:
            ratio_resize_flag = False
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if cfg['Architecture']['algorithm'] in ['SAR', 'RobustScanner']:
                if 'valid_ratio' in op[op_name]['keep_keys']:
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                else:
                    op[op_name]['keep_keys'] = ['image']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    return transforms, ratio_resize_flag


def set_device(device):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def main(cfg):
    logger = get_logger()
    global_config = cfg['Global']
    if cfg['Global']['pretrained_model'] is None:
        cfg['Global'][
            'pretrained_model'] = cfg['Global']['output_dir'] + '/best.pth'
    if cfg['Global']['infer_img'] is None:
        cfg['Global']['infer_img'] = '../iiit5k_test_image'
    # build post process
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])

    char_num = post_process_class.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    device = set_device(cfg['Global']['device'])
    model.eval()
    model.to(device=device)

    # create data ops
    transforms, ratio_resize_flag = build_rec_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)
    if ratio_resize_flag:
        ratio_resize = RatioRecTVReisze(cfg=cfg)
        ops.insert(-1, ratio_resize)
    t_sum = 0
    sample_num = 0
    max_len = cfg['Global']['max_text_length']
    text_len_time = [0 for _ in range(max_len)]
    text_len_num = [0 for _ in range(max_len)]
    for file in get_image_file_list(global_config['infer_img']):
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops)
        others = None
        if cfg['Architecture']['algorithm'] in ['SAR', 'RobustScanner']:
            valid_ratio = np.expand_dims(batch[-1], axis=0)
            others = [torch.from_numpy(valid_ratio).to(device=device)]
        images = np.expand_dims(batch[0], axis=0)
        images = torch.from_numpy(images).to(device=device)
        if sample_num == 0:
            for _ in range(100):
                with torch.no_grad():
                    preds = model(images, others)
            sample_num += 1
            continue
        with torch.no_grad():
            ts = time.time()
            preds = model(images, others)
            te = time.time()
        post_result = post_process_class(preds)

        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        'label': post_result[key][0][0],
                        'score': float(post_result[key][0][1]),
                    }
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0], int):
            # for RFLearning CNT branch
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + '\t' + str(post_result[0][1])
        t_cost = te - ts
        text_len_num[min(max_len-1, len(post_result[0][0]))] += 1
        text_len_time[min(max_len-1, len(post_result[0][0]))] += t_cost
        logger.info(f'{file}\t result: {info}, time cost: {t_cost}')
        t_sum += t_cost
        sample_num += 1

    sample_num -= 1
    print(text_len_num)
    w_avg_t_cost = []
    for l_t_cost, l_num in zip(text_len_time, text_len_num):
        if l_num != 0:
            w_avg_t_cost.append(l_t_cost / l_num)
    print(w_avg_t_cost)
    w_avg_t_cost = sum(w_avg_t_cost) / len(w_avg_t_cost)

    logger.info(
        f'Sample num: {sample_num}, Weighted Avg time cost: {t_sum/sample_num}, Avg time cost: {w_avg_t_cost}'
    )
    logger.info('success!')


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

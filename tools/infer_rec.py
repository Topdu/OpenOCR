import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tools.engine import Config
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list


class RatioRecTVReisze(object):

    def __init__(self, cfg):
        self.max_ratio = cfg['Eval']['loader'].get('max_ratio', 12)
        self.base_shape = cfg['Eval']['dataset'].get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
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


class OpenRecognizer(object):

    def __init__(self, config):
        global_config = config['Global']
        self.cfg = config
        if global_config['pretrained_model'] is None:
            global_config[
                'pretrained_model'] = global_config['output_dir'] + '/best.pth'
        # build post process
        from openrec.modeling import build_model as build_rec_model
        from openrec.postprocess import build_post_process
        from openrec.preprocess import create_operators, transform
        self.transform = transform
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        char_num = self.post_process_class.get_character_num()
        config['Architecture']['Decoder']['out_channels'] = char_num
        # print(char_num)
        self.model = build_rec_model(config['Architecture'])
        load_ckpt(self.model, config)
        # exit(0)
        self.device = set_device(global_config['device'])
        self.model.eval()
        self.model.to(device=self.device)

        transforms, ratio_resize_flag = build_rec_process(self.cfg)
        global_config['infer_mode'] = True
        self.ops = create_operators(transforms, global_config)
        if ratio_resize_flag:
            ratio_resize = RatioRecTVReisze(cfg=self.cfg)
            self.ops.insert(-1, ratio_resize)

    def __call__(self,
                 img_path=None,
                 img_numpy_list=None,
                 img_numpy=None,
                 batch_num=1):

        if img_numpy is not None:
            img_numpy_list = [img_numpy]
            num_img = 1
        elif img_path is not None:
            img_path = get_image_file_list(img_path)
            num_img = len(img_path)
        elif img_numpy_list is not None:
            num_img = len(img_numpy_list)
        else:
            raise Exception('No input image path or numpy array.')
        results = []
        for img_idx in range(num_img):
            if img_numpy_list is not None:
                img = img_numpy_list[img_idx]
                data = {'image': img}
            elif img_path is not None:
                file_name = img_path[img_idx]
                with open(file_name, 'rb') as f:
                    img = f.read()
                    data = {'image': img}
                data = self.transform(data, self.ops[:1])
            batch = self.transform(data, self.ops[1:])
            others = None
            if self.cfg['Architecture']['algorithm'] in [
                    'SAR', 'RobustScanner'
            ]:
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                others = [torch.from_numpy(valid_ratio).to(device=self.device)]
            images = np.expand_dims(batch[0], axis=0)
            images = torch.from_numpy(images).to(device=self.device)

            with torch.no_grad():
                t_start = time.time()
                preds = self.model(images, others)
                torch.cuda.synchronize()
                t_cost = time.time() - t_start
            post_result = self.post_process_class(preds)

            if img_path is not None:
                info = {
                    'file': file_name,
                    'text': post_result[0][0],
                    'score': post_result[0][1],
                    'latency': t_cost
                }
            else:
                info = {
                    'text': post_result[0][0],
                    'score': post_result[0][1],
                    'latency': t_cost
                }

            results.append(info)
        return results


def main(cfg):
    logger = get_logger()
    if cfg['Global']['infer_img'] is None:
        cfg['Global']['infer_img'] = '../iiit5k_test_image'
    model = OpenRecognizer(cfg)

    t_sum = 0
    sample_num = 0
    max_len = cfg['Global']['max_text_length']
    text_len_time = [0 for _ in range(max_len)]
    text_len_num = [0 for _ in range(max_len)]

    rec_result = model(img_path=cfg['Global']['infer_img'])

    for post_result in rec_result:
        rec_text = post_result['text']
        score = post_result['score']
        t_cost = post_result['latency']
        file = post_result['file']
        info = rec_text + '\t' + str(score)
        text_len_num[min(max_len - 1, len(rec_text))] += 1
        text_len_time[min(max_len - 1, len(rec_text))] += t_cost
        logger.info(f'{file}\t result: {info}, time cost: {t_cost}')
        t_sum += t_cost
        sample_num += 1

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

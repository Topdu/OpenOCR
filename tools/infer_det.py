from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

import cv2
import json
import torch

from opendet.modeling import build_model
from opendet.postprocess import build_post_process
from opendet.preprocess import create_operators, transform
from tools.engine import Config
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list

logger = get_logger()


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    import cv2

    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)
    logger.info('The detected Image saved in {}'.format(save_path))


def set_device(device, numId=0):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{numId}')
    else:
        device = torch.device('cpu')
    return device


class OpenDetector(object):

    def __init__(self, config, numId=0):
        from opendet.modeling import build_model as build_det_model
        from opendet.postprocess import build_post_process
        from opendet.preprocess import create_operators, transform
        self.transform = transform
        global_config = config['Global']

        # build model
        self.model = build_det_model(config['Architecture'])
        self.model.eval()
        load_ckpt(self.model, config)
        replace_batchnorm(self.model.backbone)
        self.device = set_device(config['Global']['device'], numId=numId)
        self.model.to(device=self.device)

        # create data ops
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image', 'shape']
            transforms.append(op)

        self.ops = create_operators(transforms, global_config)

        save_res_path = config['Global']['save_res_path']
        if not os.path.exists(os.path.dirname(save_res_path)):
            os.makedirs(os.path.dirname(save_res_path))

        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

    def __call__(self,
                 img_path=None,
                 img_numpy_list=None,
                 img_numpy=None,
                 batch_num=1):

        if img_numpy is not None:
            img_numpy_list = [img_numpy]
            num_img = 1
        elif img_path is not None:
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
                with open(img_path[img_idx], 'rb') as f:
                    img = f.read()
                    data = {'image': img}
                data = self.transform(data, self.ops[:1])
            batch = self.transform(data, self.ops[1:])

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images).to(device=self.device)
            with torch.no_grad():
                t_start = time.time()
                preds = self.model(images)
                t_cost = time.time() - t_start
            post_result = self.post_process_class(preds, shape_list)

            info = {'boxes': post_result[0]['points'], 'latency': t_cost}
            # print(info)
            results.append(info)
        return results


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


@torch.no_grad()
def main(config):
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])
    model.eval()
    load_ckpt(model, config)
    replace_batchnorm(model.backbone)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    device = set_device(config['Global']['device'])

    model.to(device=device)
    torch.cuda.synchronize()
    start_loop = time.time()
    with open(save_res_path, 'wb') as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info('infer_img: {}'.format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images).to(device=device)
            print(images.shape)
            # preds = model(images)
            # for i in range(1):
            t_start = time.time()
            preds = model(images)
            torch.cuda.synchronize()
            t_consume = time.time() - t_start
            print('time consume: {}'.format(t_consume))
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {'transcription': ''}
                        tmp_json['points'] = np.array(box).tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(
                        config['Global']
                        ['save_res_path']) + '/det_results_{}/'.format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {'transcription': ''}
                    tmp_json['points'] = np.array(box).tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = (
                    os.path.dirname(config['Global']['save_res_path']) +
                    '/det_results/')
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + '\t' + json.dumps(dt_boxes_json) + '\n'
            fout.write(otstr.encode())

            # onnx_file = 'dbnet_fix.onnx'
            # # x = torch.from_numpy(images)
            # torch.onnx.export(model.cpu(), images.cpu(), onnx_file, input_names=['input'], output_names=['output'], export_params=True)
            #                 #   output_names=['output'], export_params=True, dynamic_axes={'input':{0: 'batch_size', 2: 'height', 3: 'width'}})
            # import onnx
            # onnx_model = onnx.load(onnx_file)
            # onnx.checker.check_model(onnx_model)
    end_loop = time.time()
    logger.info('time total consume: {}'.format(end_loop - start_loop))

    logger.info('success!')


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

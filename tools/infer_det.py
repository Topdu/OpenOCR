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
from tools.engine import Config
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list


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


def padding_image(img, size=(640, 640)):
    """
    Padding an image using OpenCV:
    - If the image is smaller than the target size, pad it to 640x640.
    - If the image is larger than the target size, split it into multiple 640x640 images and record positions.

    :param image_path: Path to the input image.
    :param output_dir: Directory to save the output images.
    :param size: The target size for padding or splitting (default 640x640).
    :return: List of tuples containing the coordinates of the top-left corner of each cropped 640x640 image.
    """

    img_height, img_width = img.shape[:2]
    target_width, target_height = size

    # If image is smaller than target size, pad the image to 640x640

    # Calculate padding amounts (top, bottom, left, right)
    pad_top = 0
    pad_bottom = target_height - img_height
    pad_left = 0
    pad_right = target_width - img_width

    # Pad the image (white padding, border type: constant)
    padded_img = cv2.copyMakeBorder(img,
                                    pad_top,
                                    pad_bottom,
                                    pad_left,
                                    pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    # Return the padded area positions (top-left and bottom-right coordinates of the original image)
    return padded_img


def resize_image(img, size=(640, 640), over_lap=64):
    """
    Resize an image using OpenCV:
    - If the image is smaller than the target size, pad it to 640x640.
    - If the image is larger than the target size, split it into multiple 640x640 images and record positions.

    :param image_path: Path to the input image.
    :param output_dir: Directory to save the output images.
    :param size: The target size for padding or splitting (default 640x640).
    :return: List of tuples containing the coordinates of the top-left corner of each cropped 640x640 image.
    """

    img_height, img_width = img.shape[:2]
    target_width, target_height = size

    # If image is smaller than target size, pad the image to 640x640
    if img_width <= target_width and img_height <= target_height:
        # Calculate padding amounts (top, bottom, left, right)
        if img_width == target_width and img_height == target_height:
            return [img], [[0, 0, img_width, img_height]]
        padded_img = padding_image(img, size)

        # Return the padded area positions (top-left and bottom-right coordinates of the original image)
        return [padded_img], [[0, 0, img_width, img_height]]

    img_height, img_width = img.shape[:2]
    # If image is larger than or equal to target size, crop it into 640x640 tiles
    crop_positions = []
    count = 0
    cropped_img_list = []
    for top in range(0, img_height - over_lap, target_height - over_lap):
        for left in range(0, img_width - over_lap, target_width - over_lap):
            # Calculate the bottom and right boundaries for the crop
            right = min(left + target_width, img_width)
            bottom = min(top + target_height, img_height)
            if right >= img_width:
                right = img_width
                left = max(0, right - target_width)
            if bottom >= img_height:
                bottom = img_height
                top = max(0, bottom - target_height)
            # Crop the image
            cropped_img = img[top:bottom, left:right]
            if bottom - top < target_height or right - left < target_width:
                cropped_img = padding_image(cropped_img, size)
            count += 1
            cropped_img_list.append(cropped_img)

            # Record the position of the cropped image
            crop_positions.append([left, top, right, bottom])

    return cropped_img_list, crop_positions


def restore_preds(preds, crop_positions, original_size):

    restored_pred = torch.zeros((1, 1, original_size[0], original_size[1]),
                                dtype=preds.dtype,
                                device=preds.device)
    count = 0
    for cropped_pred, (left, top, right, bottom) in zip(preds, crop_positions):

        crop_height = bottom - top
        crop_width = right - left

        corp_vis_img = cropped_pred[:, :crop_height, :crop_width]
        mask = corp_vis_img > 0.3
        count += 1
        restored_pred[:, :, top:top + crop_height, left:left +
                      crop_width] += mask[:, :crop_height, :crop_width].to(
                          preds.dtype)

    return restored_pred


def draw_det_res(dt_boxes, img, img_name, save_path):
    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)


def set_device(device, numId=0):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{numId}')
    else:
        device = torch.device('cpu')
    return device


class OpenDetector(object):

    def __init__(self, config=None, numId=0):
        """
        初始化函数。

        Args:
            config (dict, optional): 配置文件，默认为None。如果为None，则使用默认配置文件。
            numId (int, optional): 设备编号，默认为0。

        Returns:
            None

        Raises:
            无
        """

        if config is None:
            config = Config('./configs/det/dbnet/repvit_db.yml').cfg

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

    def crop_infer(
        self,
        img_path=None,
        img_numpy_list=None,
        img_numpy=None,
    ):
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
            src_img_ori = data['image']
            img_height, img_width = src_img_ori.shape[:2]

            target_size = 640
            over_lap = 64
            if img_height > img_width:
                r_h = target_size * 2 - over_lap
                r_w = img_width * (target_size * 2 - over_lap) // img_height
            else:
                r_w = target_size * 2 - over_lap
                r_h = img_height * (target_size * 2 - over_lap) // img_width
            src_img = cv2.resize(src_img_ori, (r_w, r_h))
            shape_list_ori = np.array([[
                img_height, img_width,
                float(r_h) / img_height,
                float(r_w) / img_width
            ]])
            img_height, img_width = src_img.shape[:2]
            cropped_img_list, crop_positions = resize_image(src_img,
                                                            size=(target_size,
                                                                  target_size),
                                                            over_lap=over_lap)

            image_list = []
            shape_list = []
            for img in cropped_img_list:
                batch_i = self.transform({'image': img}, self.ops[-3:-1])
                image_list.append(batch_i['image'])
                shape_list.append([640, 640, 1, 1])
            images = np.array(image_list)
            shape_list = np.array(shape_list)
            images = torch.from_numpy(images).to(device=self.device)

            t_start = time.time()
            preds = self.model(images)
            torch.cuda.synchronize()
            t_cost = time.time() - t_start

            preds['maps'] = restore_preds(preds['maps'], crop_positions,
                                          (img_height, img_width))
            post_result = self.post_process_class(preds, shape_list_ori)
            info = {'boxes': post_result[0]['points'], 'elapse': t_cost}
            results.append(info)
        return results

    def __call__(self, img_path=None, img_numpy_list=None, img_numpy=None):
        """
        对输入图像进行处理，并返回处理结果。

        Args:
            img_path (str, optional): 图像文件路径。默认为 None。
            img_numpy_list (list, optional): 图像数据列表，每个元素为 numpy 数组。默认为 None。
            img_numpy (numpy.ndarray, optional): 图像数据，numpy 数组格式。默认为 None。

        Returns:
            list: 包含处理结果的列表。每个元素为一个字典，包含 'boxes' 和 'elapse' 两个键。
                'boxes' 的值为检测到的目标框点集，'elapse' 的值为处理时间。

        Raises:
            Exception: 若没有提供图像路径或 numpy 数组，则抛出异常。

        """

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

            info = {'boxes': post_result[0]['points'], 'elapse': t_cost}
            results.append(info)
        return results


@torch.no_grad()
def main(cfg):
    logger = get_logger()
    is_visualize = cfg['Global'].get('is_visualize', False)
    model = OpenDetector(cfg)

    save_res_path = cfg['Global']['output_dir']
    if not os.path.exists(save_res_path):
        os.makedirs(save_res_path)
    sample_num = 0
    with open(save_res_path + '/det_results.txt', 'wb') as fout:
        for file in get_image_file_list(cfg['Global']['infer_img']):

            preds_result = model(img_path=file)[0]
            logger.info('{} infer_img: {}, time cost: {}'.format(
                sample_num, file, preds_result['elapse']))
            boxes = preds_result['boxes']
            dt_boxes_json = []
            for box in boxes:
                tmp_json = {}
                tmp_json['points'] = np.array(box).tolist()
                dt_boxes_json.append(tmp_json)
            if is_visualize:
                src_img = cv2.imread(file)
                save_det_path = save_res_path + '/det_results/'
                draw_det_res(boxes, src_img, file, save_det_path)
                logger.info('The detected Image saved in {}'.format(
                    os.path.join(save_det_path, os.path.basename(file))))
            otstr = file + '\t' + json.dumps(dt_boxes_json) + '\n'
            logger.info('results: {}'.format(json.dumps(dt_boxes_json)))
            fout.write(otstr.encode())
            sample_num += 1

    logger.info('success!')


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

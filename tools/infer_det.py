from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
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
from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list

logger = get_logger()

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH_DET = str(root_dir / '../configs/det/dbnet/repvit_db.yml')

MODEL_NAME_DET = './openocr_det_repvit_ch.pth'  # 模型文件名称
DOWNLOAD_URL_DET = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth'  # 模型文件 URL
MODEL_NAME_DET_ONNX = './openocr_det_model.onnx'  # 模型文件名称
DOWNLOAD_URL_DET_ONNX = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_model.onnx'  # 模型文件 URL


def check_and_download_model(model_name: str, url: str):
    """
    检查预训练模型是否存在，若不存在则从指定 URL 下载到固定缓存目录。

    Args:
        model_name (str): 模型文件的名称，例如 "model.pt"
        url (str): 模型文件的下载地址

    Returns:
        str: 模型文件的完整路径
    """
    if os.path.exists(model_name):
        return model_name

    # 固定缓存路径为用户主目录下的 ".cache/openocr"
    cache_dir = Path.home() / '.cache' / 'openocr'
    model_path = cache_dir / model_name

    # 如果模型文件已存在，直接返回路径
    if model_path.exists():
        logger.info(f'Model already exists at: {model_path}')
        return str(model_path)

    # 如果文件不存在，下载模型
    logger.info(f'Model not found. Downloading from {url}...')

    # 创建缓存目录（如果不存在）
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 下载文件
        import urllib.request
        with urllib.request.urlopen(url) as response, open(model_path,
                                                           'wb') as out_file:
            out_file.write(response.read())
        logger.info(f'Model downloaded and saved at: {model_path}')
        return str(model_path)

    except Exception as e:
        logger.error(f'Error downloading the model: {e}')
        # 提示用户手动下载
        logger.error(
            f'Unable to download the model automatically. '
            f'Please download the model manually from the following URL:\n{url}\n'
            f'and save it to: {model_name} or {model_path}')
        raise RuntimeError(
            f'Failed to download the model. Please download it manually from {url} '
            f'and save it to {model_path}') from e


def replace_batchnorm(net):
    import torch
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


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
    import torch
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{numId}')
    else:
        logger.info('GPU is not available, using CPU.')
        device = torch.device('cpu')
    return device


class OpenDetector(object):

    def __init__(self,
                 config=None,
                 backend='torch',
                 onnx_model_path=None,
                 numId=0):
        """
        Args:
            config (dict, optional): 配置信息。默认为None。
            backend (str): 'torch' 或 'onnx'
            onnx_model_path (str): ONNX模型路径（仅当backend='onnx'时需要）
            numId (int, optional): 设备编号。默认为0。
        """

        if config is None:
            config = Config(DEFAULT_CFG_PATH_DET).cfg

        self._init_common(config)
        backend = backend if config['Global'].get(
            'backend', None) is None else config['Global']['backend']
        self.backend = backend
        if backend == 'torch':
            import torch
            self.torch = torch
            if config['Architecture']['algorithm'] == 'DB_mobile':
                if not os.path.exists(config['Global']['pretrained_model']):
                    config['Global'][
                        'pretrained_model'] = check_and_download_model(
                            MODEL_NAME_DET, DOWNLOAD_URL_DET)
            self._init_torch_model(config, numId)
        elif backend == 'onnx':
            from tools.infer.onnx_engine import ONNXEngine
            onnx_model_path = onnx_model_path if config['Global'].get(
                'onnx_model_path',
                None) is None else config['Global']['onnx_model_path']
            if onnx_model_path is None:
                if config['Architecture']['algorithm'] == 'DB_mobile':
                    onnx_model_path = check_and_download_model(
                        MODEL_NAME_DET_ONNX, DOWNLOAD_URL_DET_ONNX)
                else:
                    raise ValueError('ONNX模式需要指定onnx_model_path参数')
            self.onnx_det_engine = ONNXEngine(
                onnx_model_path, use_gpu=config['Global']['device'] == 'gpu')
        else:
            raise ValueError("backend参数必须是'torch'或'onnx'")

    def _init_common(self, config):
        from opendet.postprocess import build_post_process
        from opendet.preprocess import create_operators, transform
        global_config = config['Global']
        # create data ops
        self.transform = transform
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image', 'shape']
            transforms.append(op)

        self.ops = create_operators(transforms, global_config)
        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

    def _init_torch_model(self, config, numId=0):

        from opendet.modeling import build_model as build_det_model
        from tools.utils.ckpt import load_ckpt

        # build model
        self.model = build_det_model(config['Architecture'])
        self.model.eval()
        load_ckpt(self.model, config)
        if config['Architecture']['algorithm'] == 'DB_mobile':
            replace_batchnorm(self.model.backbone)
        self.device = set_device(config['Global']['device'], numId=numId)
        self.model.to(device=self.device)

    def _inference_onnx(self, images):
        # ONNX输入需要为numpy数组
        return self.onnx_det_engine.run(images)

    def __call__(self,
                 img_path=None,
                 img_numpy_list=None,
                 img_numpy=None,
                 return_mask=False,
                 **kwargs):
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
            if kwargs.get('det_input_size', None) is not None:
                data['max_sile_len'] = kwargs['det_input_size']
            batch = self.transform(data, self.ops[1:])

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            t_start = time.time()

            if self.backend == 'torch':
                images = self.torch.from_numpy(images).to(device=self.device)
                with self.torch.no_grad():
                    preds = self.model(images)
                kwargs['torch_tensor'] = True
            elif self.backend == 'onnx':
                preds_det = self._inference_onnx(images)
                preds = {'maps': preds_det[0]}
                kwargs['torch_tensor'] = False

            t_cost = time.time() - t_start
            post_result = self.post_process_class(preds, [None, shape_list],
                                                  **kwargs)

            info = {'boxes': post_result[0]['points'], 'elapse': t_cost}
            if return_mask:
                if isinstance(preds['maps'], self.torch.Tensor):
                    mask = preds['maps'].detach().cpu().numpy()
                else:
                    mask = preds['maps']
                info['mask'] = mask
            results.append(info)
        return results


def main(cfg):
    is_visualize = cfg['Global'].get('is_visualize', False)
    model = OpenDetector(cfg)

    save_res_path = './det_results/'
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
                draw_det_res(boxes, src_img, file, save_res_path)
                logger.info('The detected Image saved in {}'.format(
                    os.path.join(save_res_path, os.path.basename(file))))
            otstr = file + '\t' + json.dumps(dt_boxes_json) + '\n'
            logger.info('results: {}'.format(json.dumps(dt_boxes_json)))
            fout.write(otstr.encode())
            sample_num += 1
        logger.info(
            f"Results saved to {os.path.join(save_res_path, 'det_results.txt')}.)"
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

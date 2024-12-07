import os
from pathlib import Path
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
from tools.infer_det import replace_batchnorm

logger = get_logger()

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH_REC_SERVER = str(root_dir /
                                  '../configs/rec/svtrv2/svtrv2_ch.yml')
DEFAULT_CFG_PATH_REC = str(root_dir / '../configs/rec/svtrv2/repsvtr_ch.yml')
DEFAULT_DICT_PATH_REC = str(root_dir / './utils/ppocr_keys_v1.txt')

MODEL_NAME_REC = './openocr_repsvtr_ch.pth'  # 模型文件名称
DOWNLOAD_URL_REC = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth'  # 模型文件 URL
MODEL_NAME_REC_SERVER = './openocr_svtrv2_ch.pth'  # 模型文件名称
DOWNLOAD_URL_REC_SERVER = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth'  # 模型文件 URL


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


def set_device(device, numId=0):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{numId}')
    else:
        logger.info('GPU is not available, using CPU.')
        device = torch.device('cpu')
    return device


class OpenRecognizer(object):

    def __init__(self, config=None, mode='mobile', numId=0):
        """
        初始化方法。

        Args:
            config (dict, optional): 配置信息。默认为None。
            mode (str, optional): 模式，'server' 或 'mobile'。默认为'mobile'。
            numId (int, optional): 设备编号。默认为0。

        Returns:
            None

        Raises:
            无

        """
        if config is None:
            if mode == 'server':
                config = Config(
                    DEFAULT_CFG_PATH_REC_SERVER).cfg  # server model
                if not os.path.exists(config['Global']['pretrained_model']):
                    model_dir = check_and_download_model(
                        MODEL_NAME_REC_SERVER, DOWNLOAD_URL_REC_SERVER)
            else:
                config = Config(DEFAULT_CFG_PATH_REC).cfg  # mobile model
                if not os.path.exists(config['Global']['pretrained_model']):
                    model_dir = check_and_download_model(
                        MODEL_NAME_REC, DOWNLOAD_URL_REC)
            config['Global']['pretrained_model'] = model_dir
            config['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC
        else:
            if config['Architecture']['algorithm'] == 'SVTRv2_mobile':
                if not os.path.exists(config['Global']['pretrained_model']):
                    config['Global'][
                        'pretrained_model'] = check_and_download_model(
                            MODEL_NAME_REC, DOWNLOAD_URL_REC)
                config['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC
            elif config['Architecture']['algorithm'] == 'SVTRv2_server':
                if not os.path.exists(config['Global']['pretrained_model']):
                    config['Global'][
                        'pretrained_model'] = check_and_download_model(
                            MODEL_NAME_REC_SERVER, DOWNLOAD_URL_REC_SERVER)
                config['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC
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
        self.device = set_device(global_config['device'], numId=numId)
        self.model.eval()
        replace_batchnorm(self.model.encoder)
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
        """
        调用函数，处理输入图像，并返回识别结果。

        Args:
            img_path (str, optional): 图像文件的路径。默认为 None。
            img_numpy_list (list, optional): 包含多个图像 numpy 数组的列表。默认为 None。
            img_numpy (numpy.ndarray, optional): 单个图像的 numpy 数组。默认为 None。
            batch_num (int, optional): 每次处理的图像数量。默认为 1。

        Returns:
            list: 包含识别结果的列表，每个元素为一个字典，包含文件路径（如果有的话）、文本、分数和延迟时间。

        Raises:
            Exception: 如果没有提供图像路径或 numpy 数组，则引发异常。
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
        for start_idx in range(0, num_img, batch_num):
            batch_data = []
            batch_others = []
            batch_file_names = []

            max_width, max_height = 0, 0
            # Prepare batch data
            for img_idx in range(start_idx, min(start_idx + batch_num,
                                                num_img)):
                if img_numpy_list is not None:
                    img = img_numpy_list[img_idx]
                    data = {'image': img}
                elif img_path is not None:
                    file_name = img_path[img_idx]
                    with open(file_name, 'rb') as f:
                        img = f.read()
                        data = {'image': img}
                    data = self.transform(data, self.ops[:1])
                    batch_file_names.append(file_name)
                batch = self.transform(data, self.ops[1:])
                others = None
                if self.cfg['Architecture']['algorithm'] in [
                        'SAR', 'RobustScanner'
                ]:
                    valid_ratio = np.expand_dims(batch[-1], axis=0)
                    batch_others.append(valid_ratio)
                    # others = [torch.from_numpy(valid_ratio).to(device=self.device)]
                resized_image = batch[0]
                h, w = resized_image.shape[-2:]
                max_width = max(max_width, w)
                max_height = max(max_height, h)
                batch_data.append(batch[0])

            padded_batch_data = []
            for resized_image in batch_data:
                padded_image = np.zeros([1, 3, max_height, max_width],
                                        dtype=np.float32)
                h, w = resized_image.shape[-2:]

                # Apply padding (bottom-right padding)
                padded_image[:, :, :h, :
                             w] = resized_image  # 0 is typically used for padding
                padded_batch_data.append(padded_image)

            if batch_others:
                others = np.concatenate(batch_others, axis=0)
            else:
                others = None
            images = np.concatenate(padded_batch_data, axis=0)
            images = torch.from_numpy(images).to(device=self.device)

            with torch.no_grad():
                t_start = time.time()
                preds = self.model(images, others)
                t_cost = time.time() - t_start
            post_results = self.post_process_class(preds)

            for i, post_result in enumerate(post_results):
                if img_path is not None:
                    info = {
                        'file': batch_file_names[i],
                        'text': post_result[0],
                        'score': post_result[1],
                        'elapse': t_cost
                    }
                else:
                    info = {
                        'text': post_result[0],
                        'score': post_result[1],
                        'elapse': t_cost
                    }
                results.append(info)

        return results


def main(cfg):
    model = OpenRecognizer(cfg)

    save_res_path = './rec_results/'
    if not os.path.exists(save_res_path):
        os.makedirs(save_res_path)

    t_sum = 0
    sample_num = 0
    max_len = cfg['Global']['max_text_length']
    text_len_time = [0 for _ in range(max_len)]
    text_len_num = [0 for _ in range(max_len)]

    sample_num = 0
    with open(save_res_path + '/rec_results.txt', 'wb') as fout:
        for file in get_image_file_list(cfg['Global']['infer_img']):
            preds_result = model(img_path=file, batch_num=1)[0]
            rec_text = preds_result['text']
            score = preds_result['score']
            t_cost = preds_result['elapse']
            info = rec_text + '\t' + str(score)
            text_len_num[min(max_len - 1, len(rec_text))] += 1
            text_len_time[min(max_len - 1, len(rec_text))] += t_cost
            logger.info(
                f'{sample_num} {file}\t result: {info}, time cost: {t_cost}')
            otstr = file + '\t' + info + '\n'
            t_sum += t_cost
            fout.write(otstr.encode())
            sample_num += 1
        logger.info(
            f"Results saved to {os.path.join(save_res_path, 'rec_results.txt')}.)"
        )

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

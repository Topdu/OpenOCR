import os
from pathlib import Path
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np
from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list

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
MODEL_NAME_REC_ONNX = './openocr_rec_model.onnx'  # 模型文件名称
DOWNLOAD_URL_REC_ONNX = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_rec_model.onnx'  # 模型文件 URL


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

        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        self.F = F
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
        resized_image = self.F.resize(img, (imgH, resized_w),
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
    import torch
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{numId}')
    else:
        logger.info('GPU is not available, using CPU.')
        device = torch.device('cpu')
    return device


class OpenRecognizer:

    def __init__(self,
                 config=None,
                 mode='mobile',
                 backend='torch',
                 onnx_model_path=None,
                 numId=0):
        """
        Args:
            config (dict, optional): 配置信息。默认为None。
            mode (str, optional): 模式，'server' 或 'mobile'。默认为'mobile'。
            backend (str): 'torch' 或 'onnx'
            onnx_model_path (str): ONNX模型路径（仅当backend='onnx'时需要）
            numId (int, optional): 设备编号。默认为0。
        """

        if config is None:
            config_file = DEFAULT_CFG_PATH_REC_SERVER if mode == 'server' else DEFAULT_CFG_PATH_REC
            config = Config(config_file).cfg
        self.cfg = config
        # 公共初始化
        self._init_common()
        backend = backend if config['Global'].get(
            'backend', None) is None else config['Global']['backend']
        self.backend = backend
        if backend == 'torch':
            import torch
            self.torch = torch
            self._init_torch_model(numId)
        elif backend == 'onnx':
            from tools.infer.onnx_engine import ONNXEngine
            onnx_model_path = onnx_model_path if config['Global'].get(
                'onnx_model_path',
                None) is None else config['Global']['onnx_model_path']
            if not onnx_model_path:
                if self.cfg['Architecture']['algorithm'] == 'SVTRv2_mobile':
                    onnx_model_path = check_and_download_model(
                        MODEL_NAME_REC_ONNX, DOWNLOAD_URL_REC_ONNX)
                else:
                    raise ValueError('ONNX模式需要指定onnx_model_path参数')
            self.onnx_rec_engine = ONNXEngine(
                onnx_model_path, use_gpu=config['Global']['device'] == 'gpu')
        else:
            raise ValueError("backend参数必须是'torch'或'onnx'")

    def _init_common(self):
        # 初始化公共组件
        from openrec.postprocess import build_post_process
        from openrec.preprocess import create_operators, transform
        self.transform = transform
        # 构建预处理流程
        algorithm_name = self.cfg['Architecture']['algorithm']
        if algorithm_name in ['SVTRv2_mobile', 'SVTRv2_server']:
            self.cfg['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC
        self.post_process_class = build_post_process(self.cfg['PostProcess'],
                                                     self.cfg['Global'])
        char_num = self.post_process_class.get_character_num()
        self.cfg['Architecture']['Decoder']['out_channels'] = char_num
        transforms, ratio_resize_flag = build_rec_process(self.cfg)
        self.ops = create_operators(transforms, self.cfg['Global'])
        if ratio_resize_flag:
            ratio_resize = RatioRecTVReisze(cfg=self.cfg)
            self.ops.insert(-1, ratio_resize)

    def _init_torch_model(self, numId):
        from tools.utils.ckpt import load_ckpt
        from tools.infer_det import replace_batchnorm
        # PyTorch专用初始化
        algorithm_name = self.cfg['Architecture']['algorithm']
        if algorithm_name in ['SVTRv2_mobile', 'SVTRv2_server']:
            if not os.path.exists(self.cfg['Global']['pretrained_model']):
                pretrained_model = check_and_download_model(
                    MODEL_NAME_REC, DOWNLOAD_URL_REC
                ) if algorithm_name == 'SVTRv2_mobile' else check_and_download_model(
                    MODEL_NAME_REC_SERVER, DOWNLOAD_URL_REC_SERVER)
                self.cfg['Global']['pretrained_model'] = pretrained_model

        from openrec.modeling import build_model as build_rec_model

        self.model = build_rec_model(self.cfg['Architecture'])
        load_ckpt(self.model, self.cfg)

        self.device = set_device(self.cfg['Global']['device'], numId)
        self.model.to(self.device)
        self.model.eval()
        if algorithm_name == 'SVTRv2_mobile':
            replace_batchnorm(self.model.encoder)

    def _inference_onnx(self, images):
        # ONNX输入需要为numpy数组
        return self.onnx_rec_engine.run(images)

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

                resized_image = batch[0] if isinstance(
                    batch[0], np.ndarray) else batch[0].numpy()
                h, w = resized_image.shape[-2:]
                max_width = max(max_width, w)
                max_height = max(max_height, h)
                batch_data.append(batch[0])

            padded_batch = np.zeros(
                (len(batch_data), 3, max_height, max_width), dtype=np.float32)
            for i, img in enumerate(batch_data):
                h, w = img.shape[-2:]
                padded_batch[i, :, :h, :w] = img

            if batch_others:
                others = np.concatenate(batch_others, axis=0)
            else:
                others = None
            t_start = time.time()
            if self.backend == 'torch':
                images = self.torch.from_numpy(padded_batch).to(
                    device=self.device)
                with self.torch.no_grad():
                    preds = self.model(images, others)  # bs, len, num_classes
                torch_tensor = True
            elif self.backend == 'onnx':
                # ONNX推理
                preds = self._inference_onnx(padded_batch)
                preds = preds[0]  # bs, len, num_classes
                torch_tensor = False
            t_cost = time.time() - t_start
            post_results = self.post_process_class(preds,
                                                   torch_tensor=torch_tensor)
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

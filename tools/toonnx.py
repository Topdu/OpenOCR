import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import torch

from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger


def to_onnx(model, dummy_input, dynamic_axes, sava_path='model.onnx'):
    input_axis_name = ['batch_size', 'channel', 'in_width', 'int_height']
    output_axis_name = ['batch_size', 'channel', 'out_width', 'out_height']
    torch.onnx.export(
        model.to('cpu'),
        dummy_input,
        sava_path,
        input_names=['input'],
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {axis: input_axis_name[axis]
                      for axis in dynamic_axes},
            'output': {axis: output_axis_name[axis]
                       for axis in dynamic_axes},
        },
    )


def main(cfg):
    _cfg = cfg.cfg
    logger = get_logger()
    global_config = _cfg['Global']

    export_dir = global_config.get('export_dir', '')

    if _cfg['Architecture']['algorithm'] == 'SVTRv2_mobile':
        from tools.infer_rec import OpenRecognizer
        model = OpenRecognizer(_cfg).model
        dynamic_axes = [0, 3]
        dummy_input = torch.randn([1, 3, 48, 320], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_rec')
        save_path = os.path.join(export_dir, 'rec_model.onnx')
    if _cfg['Architecture']['algorithm'] == 'DB_mobile':
        from tools.infer_det import OpenDetector
        model = OpenDetector(_cfg).model
        dynamic_axes = [0, 2, 3]
        dummy_input = torch.randn([1, 3, 960, 960], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_det')
        save_path = os.path.join(export_dir, 'det_model.onnx')

    os.makedirs(export_dir, exist_ok=True)
    to_onnx(model, dummy_input, dynamic_axes, save_path)
    logger.info(f'finish export model to {save_path}')


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg)

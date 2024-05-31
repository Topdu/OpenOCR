import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import torch

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from tools.engine import Config
from tools.infer_rec import build_rec_process
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
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


def export_single_model(model: torch.nn.Module, _cfg, export_dir,
                        export_config, logger, type):
    for layer in model.modules():
        if hasattr(layer, 'rep') and not getattr(layer, 'is_repped'):
            layer.rep()
    os.makedirs(export_dir, exist_ok=True)

    export_cfg = {'PostProcess': _cfg['PostProcess']}
    export_cfg['Transforms'] = build_rec_process(_cfg)

    cfg.save(os.path.join(export_dir, 'config.yaml'), export_cfg)

    dummy_input = torch.randn(*export_config['export_shape'], device='cpu')
    if type == 'script':
        save_path = os.path.join(export_dir, 'model.pt')
        trace_model = torch.jit.trace(model, dummy_input, strict=False)
        torch.jit.save(trace_model, save_path)
    elif type == 'onnx':
        save_path = os.path.join(export_dir, 'model.onnx')
        to_onnx(model, dummy_input, export_config.get('dynamic_axes', []),
                save_path)
    else:
        raise NotImplementedError
    logger.info(f'finish export model to {save_path}')


def main(cfg, type):
    _cfg = cfg.cfg
    logger = get_logger()
    global_config = _cfg['Global']
    export_config = _cfg['Export']
    # build post process
    post_process_class = build_post_process(_cfg['PostProcess'])
    char_num = len(getattr(post_process_class, 'character'))
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(_cfg['Architecture'])

    load_ckpt(model, _cfg)
    model.eval()

    export_dir = export_config.get('export_dir', '')
    if not export_dir:
        export_dir = os.path.join(global_config.get('output_dir', 'output'),
                                  'export')

    if _cfg['Architecture']['algorithm'] in ['Distillation'
                                             ]:  # distillation model
        _cfg['PostProcess'][
            'name'] = post_process_class.__class__.__base__.__name__
        for model_name in model.model_list:
            sub_model_save_path = os.path.join(export_dir, model_name)
            export_single_model(
                model.model_list[model_name],
                _cfg,
                sub_model_save_path,
                export_config,
                logger,
                type,
            )
    else:
        export_single_model(model, _cfg, export_dir, export_config, logger,
                            type)


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--type',
                        type=str,
                        default='onnx',
                        help='type of export')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg, FLAGS['type'])

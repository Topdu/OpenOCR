import os

import gradio as gr
import numpy as np
import torch

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from tools.engine import Config
from tools.utils.ckpt import load_ckpt


def build_rec_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        # TODO
        elif op_name in ['DecodeImage']:
            op[op_name]['gradio_infer_mode'] = True

        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if cfg['Architecture']['algorithm'] == 'SRN':
                op[op_name]['keep_keys'] = [
                    'image',
                    'encoder_word_pos',
                    'gsrm_word_pos',
                    'gsrm_slf_attn_bias1',
                    'gsrm_slf_attn_bias2',
                ]
            elif cfg['Architecture']['algorithm'] == 'SAR':
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif cfg['Architecture']['algorithm'] == 'RobustScanner':
                op[op_name]['keep_keys'] = [
                    'image', 'valid_ratio', 'word_positons'
                ]
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    return transforms


def get_all_file_names_including_subdirs(dir_path):
    all_file_names = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            all_file_names.append(os.path.join(root, file_name))

    file_names_only = [os.path.basename(file) for file in all_file_names]
    return file_names_only


root_directory = './configs/rec'
yml_Config = get_all_file_names_including_subdirs(root_directory)


def find_file_in_current_dir_and_subdirs(file_name):
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            relative_path = os.path.join(root, file_name)
            return relative_path


def predict(input_image, Model_type, OCR_type):

    path = find_file_in_current_dir_and_subdirs(Model_type)

    cfg = Config(path).cfg
    post_process_class = build_post_process(cfg['PostProcess'])
    global_config = cfg['Global']
    char_num = len(getattr(post_process_class, 'character'))
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.eval()

    transforms = build_rec_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)
    data = {'image': input_image}
    batch = transform(data, ops)
    others = None
    images = np.expand_dims(batch[0], axis=0)
    images = torch.from_numpy(images)
    with torch.no_grad():
        preds = model(images, others)
    post_result = post_process_class(preds)
    return post_result[0][0], post_result[0][1]


if __name__ == '__main__':

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input Image')

                # TODO
                OCR_type = gr.Radio(['STR', 'STD', 'E2E'], label='模型类别')

                Model_type = gr.Dropdown(choices=yml_Config, label='现有模型配置文件')

                downstream = gr.Button('识别结果')

            with gr.Column(scale=1):

                # TODO
                img_output = gr.Image(label='图片识别结果')

                output = gr.Textbox(label='文字识别结果')
                confidence = gr.Textbox(label='置信度')

            downstream.click(
                fn=predict,
                inputs=[
                    input_image,
                    Model_type,
                    OCR_type,
                ],
                outputs=[
                    output,
                    confidence,
                    # TODO img_output,
                ])

    demo.launch(debug=True)

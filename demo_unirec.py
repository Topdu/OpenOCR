import os

import torch
import gradio as gr  # gradio==4.20.0
import numpy as np
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from tools.engine.config import Config
from tools.utils.ckpt import load_ckpt
from tools.infer_rec import build_rec_process


def set_device(device):
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:0')
    else:
        device = torch.device('cpu')
    return device


cfg = Config('configs/rec/unirec/focalsvtr_ardecoder_unirec.yml')
cfg = cfg.cfg
global_config = cfg['Global']

# build post process
post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])

from openrec.modeling.transformers_modeling.modeling_unirec import UniRecForConditionalGenerationNew
from openrec.modeling.transformers_modeling.configuration_unirec import UniRecConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./configs/rec/unirec/unirec_100m')
cfg_model = UniRecConfig.from_pretrained('./configs/rec/unirec/unirec_100m')
# cfg_model._attn_implementation = "flash_attention_2"
cfg_model._attn_implementation = 'eager'

model = UniRecForConditionalGenerationNew(config=cfg_model)
load_ckpt(model, cfg)
device = set_device(cfg['Global']['device'])
model.eval()
model.to(device=device)

transforms, ratio_resize_flag = build_rec_process(cfg)
ops = create_operators(transforms, global_config)


def process_image(input_image):
    data = {'image': input_image}
    batch = transform(data, ops[1:])
    images = np.expand_dims(batch[0], axis=0)
    images = torch.from_numpy(images).to(device=device)
    with torch.no_grad():
        inputs = {
            'pixel_values': images,
            'input_ids': None,
            'attention_mask': None
        }
        preds = model.generate(**inputs)
        res = tokenizer.batch_decode(preds, skip_special_tokens=False)
        res[0] = res[0].replace(' ', '').replace('Ġ', ' ').replace(
            'Ċ', '\n').replace('<|bos|>',
                               '').replace('<|eos|>',
                                           '').replace('<|pad|>', '')

    rec_results = res[0]
    return rec_results


def list_image_paths(directory):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                relative_path = os.path.relpath(os.path.join(root, file),
                                                directory)
                full_path = os.path.join(directory, relative_path)
                image_paths.append(full_path)
    image_paths = sorted(image_paths)
    return image_paths


e2e_img_example = list_image_paths('./unirec_100m/demo_imgs')

# 创建Gradio界面
with gr.Blocks(title='文本-公式识别系统') as demo:
    gr.HTML("""
            <h1 style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR">UniRec: Unified Text and Formula Recognition Across Granularities</a></h1>
            <p style='text-align: center;'>统一多粒度文本与公式识别模型 （由<a href="https://fvl.fudan.edu.cn">FVL实验室</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a> 创建）</p>
            <p style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR/docs/unirec.md">[本地GPU部署]</a>获取快速识别</p>"""
            )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label='上传图片', type='pil')
            process_btn = gr.Button('开始处理', variant='primary')

        with gr.Column():

            edit_box = gr.Textbox(
                label='识别结果（使用LaTeX编译器渲染公式）',
                lines=10,
                placeholder='在这里编辑Markdown内容...',
                interactive=True,
                show_copy_button=True,
            )

    examples = gr.Examples(examples=e2e_img_example,
                           inputs=image_input,
                           label='示例图片')

    # 初始处理流程
    process_btn.click(fn=process_image, inputs=image_input, outputs=edit_box)

    # 实时渲染逻辑
    edit_box.change(fn=lambda x: x, inputs=edit_box)

# 启动服务 server_name="10.176.42.28"
demo.launch()

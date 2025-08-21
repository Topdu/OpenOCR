import gradio as gr
import torch
from threading import Thread

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
from transformers import AutoTokenizer, TextIteratorStreamer

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


# --- 2. 定义流式生成函数 ---
def stream_chat_with_image(input_image, history):
    if input_image is None:
        yield history + [('🖼️(空)', '请先上传一张图片。')]
        return

    # 创建 TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=False)

    data = {'image': input_image}
    batch = transform(data, ops[1:])
    images = np.expand_dims(batch[0], axis=0)
    images = torch.from_numpy(images).to(device=device)
    inputs = {
        'pixel_values': images,
        'input_ids': None,
        'attention_mask': None
    }
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
    # 后台线程运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # 流式输出
    generated_text = ''
    history = history + [('🖼️(图片)', '')]
    for new_text in streamer:
        new_text = new_text.replace(' ', '').replace('Ġ', ' ').replace(
            'Ċ', '\n').replace('<|bos|>',
                               '').replace('<|eos|>',
                                           '').replace('<|pad|>', '')
        generated_text += new_text
        history[-1] = ('🖼️(图片)', generated_text)
        yield history


# --- 3. Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
            <h1 style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR">UniRec: Unified Text and Formula Recognition Across Granularities</a></h1>
            <p style='text-align: center;'>统一多粒度文本与公式识别模型 （由<a href="https://fvl.fudan.edu.cn">FVL实验室</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a> 创建）</p>
            <p style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR/blob/openocr_svtrv2/docs/unirec.md">[本地GPU部署]</a>获取快速识别体验</p>"""
            )
    gr.Markdown('上传一张图片，系统会自动识别文本和公式。')
    with gr.Row():
        with gr.Column(scale=1):  # 左侧竖排：图片 + 清空按钮
            image_input = gr.Image(label='上传图片 or 粘贴截图', type='pil')
            clear = gr.ClearButton([image_input],
                                   value='清空')  # 先挂载到 image_input
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label='结果（请使用LaTeX编译器渲染公式）',
                                 show_copy_button=True,
                                 height='auto')
    # 再把 clear 绑定 chatbot 一起清理
    clear.add([chatbot])
    # 上传后触发
    image_input.upload(stream_chat_with_image, [image_input, chatbot], chatbot)

# --- 4. 启动应用 ---
if __name__ == '__main__':
    demo.queue().launch(share=True)

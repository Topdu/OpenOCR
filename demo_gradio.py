# @Author: OpenOCR
# @Contact: 784990967@qq.com
import os
import gradio as gr  # gradio==4.20.0

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
import cv2
import numpy as np
import json
import time
from PIL import Image
from tools.infer_e2e import OpenOCR, check_and_download_font, draw_ocr_box_txt


def initialize_ocr(model_type, drop_score):
    return OpenOCR(mode=model_type, drop_score=drop_score)


# Default model type
model_type = 'mobile'
drop_score = 0.4
text_sys = initialize_ocr(model_type, drop_score)

# warm up 5 times
if True:
    img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    for i in range(5):
        res = text_sys(img_numpy=img)

font_path = './simfang.ttf'
font_path = check_and_download_font(font_path)


def main(input_image,
         model_type_select,
         det_input_size_textbox=960,
         rec_drop_score=0.4,
         mask_thresh=0.3,
         box_thresh=0.6,
         unclip_ratio=1.5,
         det_score_mode='slow'):
    global text_sys, model_type

    # Update OCR model if the model type changes
    if model_type_select != model_type:
        model_type = model_type_select
        text_sys = initialize_ocr(model_type, rec_drop_score)

    img = input_image[:, :, ::-1]
    starttime = time.time()
    results, time_dict, mask = text_sys(
        img_numpy=img,
        return_mask=True,
        det_input_size=int(det_input_size_textbox),
        thresh=mask_thresh,
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
        score_mode=det_score_mode)
    elapse = time.time() - starttime
    save_pred = json.dumps(results[0], ensure_ascii=False)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = [res['points'] for res in results[0]]
    txts = [res['transcription'] for res in results[0]]
    scores = [res['score'] for res in results[0]]
    draw_img = draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores,
        drop_score=rec_drop_score,
        font_path=font_path,
    )
    mask = mask[0, 0, :, :] > mask_thresh
    return save_pred, elapse, draw_img, mask.astype('uint8') * 255


def get_all_file_names_including_subdirs(dir_path):
    all_file_names = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            all_file_names.append(os.path.join(root, file_name))

    file_names_only = [os.path.basename(file) for file in all_file_names]
    return file_names_only


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


def find_file_in_current_dir_and_subdirs(file_name):
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            relative_path = os.path.join(root, file_name)
            return relative_path


e2e_img_example = list_image_paths('./OCR_e2e_img')

if __name__ == '__main__':
    css = '.image-container img { width: 100%; max-height: 320px;}'

    with gr.Blocks(css=css) as demo:
        gr.HTML("""
                <h1 style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR">OpenOCR</a></h1>
                <p style='text-align: center;'>准确高效的通用 OCR 系统 （由<a href="https://fvl.fudan.edu.cn">FVL实验室</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a> 创建） <a href="https://github.com/Topdu/OpenOCR/tree/main?tab=readme-ov-file#quick-start">[本地快速部署]</a></p>"""
                )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input image',
                                       elem_classes=['image-container'])

                examples = gr.Examples(examples=e2e_img_example,
                                       inputs=input_image,
                                       label='Examples')
                downstream = gr.Button('Run')

                # 添加参数调节组件
                with gr.Column():
                    with gr.Row():
                        det_input_size_textbox = gr.Number(
                            label='Detection Input Size',
                            value=960,
                            info='检测网络输入尺寸的最长边，默认为960。')
                        det_score_mode_dropdown = gr.Dropdown(
                            ['slow', 'fast'],
                            value='slow',
                            label='Detection Score Mode',
                            info='文本框的置信度计算模式，默认为 slow。slow 模式计算速度较慢，但准确度较高。fast 模式计算速度较快，但准确度较低。'
                        )
                    with gr.Row():
                        rec_drop_score_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.4,
                            step=0.01,
                            label='Recognition Drop Score',
                            info='识别置信度阈值，默认值为0.4。低于该阈值的识别结果和对应的文本框被丢弃。')
                        mask_thresh_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.3,
                            step=0.01,
                            label='Mask Threshold',
                            info='Mask 阈值，用于二值化 mask，默认值为0.3。如果存在文本截断时，请调低该值。')
                    with gr.Row():
                        box_thresh_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.6,
                            step=0.01,
                            label='Box Threshold',
                            info='文本框置信度阈值，默认值为0.6。如果存在文本被漏检时，请调低该值。')
                        unclip_ratio_slider = gr.Slider(
                            1.5,
                            2.0,
                            value=1.5,
                            step=0.05,
                            label='Unclip Ratio',
                            info='文本框解析时的膨胀系数，默认值为1.5。值越大文本框越大。')

                    # 模型选择组件
                    model_type_dropdown = gr.Dropdown(
                        ['mobile', 'server'],
                        value='mobile',
                        label='Model Type',
                        info='选择 OCR 模型类型：高效率模型mobile，高精度模型server。')

            with gr.Column(scale=1):
                img_mask = gr.Image(label='mask',
                                    interactive=False,
                                    elem_classes=['image-container'])
                img_output = gr.Image(label=' ',
                                      interactive=False,
                                      elem_classes=['image-container'])

                output = gr.Textbox(label='Result')
                confidence = gr.Textbox(label='Latency')

            downstream.click(fn=main,
                             inputs=[
                                 input_image, model_type_dropdown,
                                 det_input_size_textbox, rec_drop_score_slider,
                                 mask_thresh_slider, box_thresh_slider,
                                 unclip_ratio_slider, det_score_mode_dropdown
                             ],
                             outputs=[
                                 output,
                                 confidence,
                                 img_output,
                                 img_mask,
                             ])

    demo.launch(share=True)

import os
import gradio as gr  # gradio==4.20.0

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
import cv2
import numpy as np
import json
import time
from PIL import Image
from tools.infer_e2e import OpenOCR, check_and_download_font, draw_ocr_box_txt

drop_score = 0.01
text_sys = OpenOCR(drop_score=drop_score)
# warm up 5 times
if True:
    img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    for i in range(5):
        res = text_sys(img_numpy=img)
font_path = './simfang.ttf'
check_and_download_font(font_path)


def main(input_image):
    # for idx, image_file in enumerate(image_file_list):
    img = input_image[:, :, ::-1]
    starttime = time.time()
    results, time_dict, mask = text_sys(img_numpy=img, return_mask=True)
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
        drop_score=drop_score,
        font_path=font_path,
    )
    mask = mask[0, 0, :, :] > 0.3
    return save_pred, elapse, draw_img, mask.astype('uint8') * 255


def get_all_file_names_including_subdirs(dir_path):
    all_file_names = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            all_file_names.append(os.path.join(root, file_name))

    file_names_only = [os.path.basename(file) for file in all_file_names]
    return file_names_only


def list_image_paths(directory):
    # 定义支持的图片扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # 列表来存储图片的相对路径
    image_paths = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                # 获取相对路径
                relative_path = os.path.relpath(os.path.join(root, file),
                                                directory)
                full_path = os.path.join(directory, relative_path)
                image_paths.append(full_path)

    return image_paths


def find_file_in_current_dir_and_subdirs(file_name):
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            relative_path = os.path.join(root, file_name)
            return relative_path


def predict1(input_image, Model_type, OCR_type):
    if OCR_type == 'E2E':
        return 11111, 'E2E', input_image
    elif OCR_type == 'STR':
        return 11111, 'STR', input_image
    else:
        return 11111, 'STD', input_image


e2e_img_example = list_image_paths('./OCR_e2e_img')

if __name__ == '__main__':
    css = '.image-container img { width: 100%; max-height: 320px;}'

    with gr.Blocks(css=css) as demo:
        gr.HTML("""
                <h1 style='text-align: center;'>OpenOCR</h1>""")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input image',
                                       elem_classes=['image-container'])

                # examples = gr.Examples(examples=rec_img_example,
                #     inputs=input_image,label = "文本识别示例图片")

                examples = gr.Examples(examples=e2e_img_example,
                                       inputs=input_image,
                                       label='示例图片')

                # OCR_type = gr.Radio(['STR', 'STD', 'E2E'], label='模型类别')

                # Model_type = gr.Dropdown(choices=yml_Config, label='现有模型配置文件')

                downstream = gr.Button('Run')

                # OCR_type.change(
                #     fn=update_examples,
                #     inputs=OCR_type,
                #     outputs=examples,
                # )

            with gr.Column(scale=1):
                img_mask = gr.Image(label='mask',
                                    interactive=False,
                                    elem_classes=['image-container'])
                img_output = gr.Image(label=' ',
                                      interactive=False,
                                      elem_classes=['image-container'])

                output = gr.Textbox(label='识别结果')
                confidence = gr.Textbox(label='耗时')

            downstream.click(
                fn=main,
                inputs=[
                    input_image,
                    # Model_type,
                    # OCR_type,
                ],
                outputs=[
                    output,
                    confidence,
                    img_output,
                    img_mask,
                ])

    demo.launch(share=True)

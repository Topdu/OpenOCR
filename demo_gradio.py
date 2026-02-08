# @Author: OpenOCR
# @Contact: 784990967@qq.com
import os
import gradio as gr

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
import cv2
import numpy as np
import json
import time
from PIL import Image
from pathlib import Path

from tools.infer_e2e import OpenOCRE2E, check_and_download_font, draw_ocr_box_txt
from tools.download_example_images import get_example_images_path


def initialize_ocr(model_type, drop_score):
    return OpenOCRE2E(mode=model_type, drop_score=drop_score, backend='onnx')


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


# Get example images path and download if necessary
example_img_dir = get_example_images_path(demo_type='ocr')
e2e_img_example = list_image_paths(example_img_dir)


def launch_demo(share=False, server_port=7860, server_name='0.0.0.0'):
    """Launch OpenOCR Gradio demo with default configuration.

    Args:
        share: Whether to create a public share link (default: False)
        server_port: Server port (default: 7860)
        server_name: Server name (default: '0.0.0.0')

    Returns:
        gr.Blocks: Gradio demo instance
    """
    custom_css = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
}
.app-header {
    text-align: center;
    max-width: 1200px;
    margin: 20px auto !important;
    padding: 20px;
}
.app-header h1 {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 10px;
}
.app-header p {
    font-size: 1.1em;
    opacity: 0.7;
    line-height: 1.6;
}
.quick-links {
    text-align: center;
    padding: 12px 0;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    margin: 16px auto;
    max-width: 1200px;
    background: var(--background-fill-secondary);
}
.quick-links a {
    margin: 0 16px;
    font-size: 15px;
    font-weight: 600;
    color: var(--link-text-color);
    text-decoration: none;
    transition: all 0.3s ease;
}
.quick-links a:hover {
    opacity: 0.8;
    text-decoration: underline;
}
.image-container img {
    max-width: 100%;
    max-height: 480px;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}
.upload-section {
    border: 2px dashed var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
    background: var(--background-fill-secondary);
    transition: all 0.3s ease;
}
.upload-section:hover {
    border-color: var(--color-accent);
    background: var(--background-fill-primary);
}
.gradio-button-primary {
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.gradio-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-drop-lg) !important;
}
"""

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div class="app-header">
            <h1><a href="https://github.com/Topdu/OpenOCR">OpenOCR</a></h1>
            <p>Accurate and Efficient General OCR System (built by <a href="https://fvl.fudan.edu.cn">FVL Lab</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a>)</p>
        </div>
        <div class="quick-links">
            <a href="https://github.com/Topdu/OpenOCR" target="_blank">📖 GitHub</a>
            <a href="https://github.com/Topdu/OpenOCR/tree/main?tab=readme-ov-file#quick-start" target="_blank">🚀 Quick Start</a>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input image',
                                       elem_classes=['image-container'])

                examples = gr.Examples(examples=e2e_img_example,
                                       inputs=input_image,
                                       label='Examples')
                downstream = gr.Button('🚀 Run Recognition', variant='primary')

                # Parameter adjustment components
                with gr.Column():
                    with gr.Row():
                        det_input_size_textbox = gr.Number(
                            label='Detection Input Size',
                            value=960,
                        info='Max side length of detection input, default 960.')
                        det_score_mode_dropdown = gr.Dropdown(
                            ['slow', 'fast'],
                            value='slow',
                            label='Detection Score Mode',
                        info=
                            'Confidence score mode for text boxes, default slow. Slow mode is more accurate but slower. Fast mode is faster but less accurate.'
                        )
                    with gr.Row():
                        rec_drop_score_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.4,
                            step=0.01,
                            label='Recognition Drop Score',
                        info='Recognition confidence threshold, default 0.4. Results below this threshold will be discarded.')
                        mask_thresh_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.3,
                            step=0.01,
                            label='Mask Threshold',
                        info='Mask threshold for binarization, default 0.3. Lower this value if text is truncated.')
                    with gr.Row():
                        box_thresh_slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.6,
                            step=0.01,
                            label='Box Threshold',
                        info='Text box confidence threshold, default 0.6. Lower this value if text boxes are missed.')
                        unclip_ratio_slider = gr.Slider(
                            1.5,
                            2.0,
                            value=1.5,
                            step=0.05,
                            label='Unclip Ratio',
                        info='Expansion ratio for text box parsing, default 1.5. Larger values produce larger text boxes.')

                    model_type_dropdown = gr.Dropdown(
                        ['mobile', 'server'],
                        value='mobile',
                        label='Model Type',
                        info='Select OCR model type: mobile for efficiency, server for accuracy.')

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
    allowed_path = str(Path.home() / '.cache' / 'openocr')
    demo.launch(share=share, server_port=server_port, server_name=server_name, allowed_paths=[allowed_path])
    return demo


if __name__ == '__main__':
    launch_demo(share=False)

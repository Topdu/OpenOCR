import os
import gradio as gr
from PIL import Image

from tools.infer_doc import OpenDoc
from tools.utils.logging import get_logger

logger = get_logger(name='opendoc_gradio')

# Initialize the pipeline
# Note: Using gpuId=-1 for CPU or 0 for the first GPU.
# You can change this based on your environment.
pipeline = None


def get_pipeline(gpu_id):
    global pipeline
    if pipeline is None:
        logger.info(
            f"Initializing OpenDoc pipeline on {'GPU ' + str(gpu_id) if gpu_id >= 0 else 'CPU'}..."
        )
        pipeline = OpenDoc(gpuId=gpu_id)
    return pipeline


# Ensure pipeline is initialized
try:
    current_pipeline = get_pipeline(0)
except Exception as e:
    raise e

import uuid


def process_image(image):
    if image is None:
        return None, '', '', None

    # Create a unique directory for this request to store files for download
    output_base_dir = 'gradio_outputs'
    os.makedirs(output_base_dir, exist_ok=True)
    request_id = str(uuid.uuid4())
    tmp_dir = os.path.join(output_base_dir, request_id)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        tmp_img_path = os.path.join(tmp_dir, 'input.jpg')
        image.save(tmp_img_path)

        # Predict
        output = list(
            current_pipeline.predict(tmp_img_path,
                                     use_doc_orientation_classify=False,
                                     use_doc_unwarping=False))
        if not output:
            return None, 'No results found.', '', None

        res = output[0]

        # Save results
        res.save_to_img(tmp_dir)
        res.save_to_markdown(tmp_dir, pretty=True)
        res.save_to_json(tmp_dir)

        # Find the saved files
        vis_img = None
        vis_img_path = None
        for f in os.listdir(tmp_dir):
            if f.endswith(('_res.jpg', '_res.png')):
                vis_img_path = os.path.join(tmp_dir, f)
                vis_img = Image.open(vis_img_path)
                break

        markdown_content = ''
        md_file_path = None
        for f in os.listdir(tmp_dir):
            if f.endswith('.md'):
                md_file_path = os.path.join(tmp_dir, f)
                with open(md_file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                break

        json_content = ''
        json_file_path = None
        for f in os.listdir(tmp_dir):
            if f.endswith('.json'):
                json_file_path = os.path.join(tmp_dir, f)
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    json_content = file.read()
                break

        # Prepare files for download
        download_files = []
        if md_file_path:
            download_files.append(md_file_path)
        if json_file_path:
            download_files.append(json_file_path)

        return vis_img, markdown_content, json_content, download_files, markdown_content

    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        return None, f'Error during prediction: {str(e)}', '', None, ''


# Define the Gradio Interface
def create_demo():
    with gr.Blocks(title='OpenDoc-0.1B Demo') as demo:
        gr.Markdown(
            '# 🚀 OpenDoc-0.1B: Ultra-Lightweight Document Parsing System')
        gr.Markdown(
            'OpenDoc-0.1B is an ultra-lightweight (0.1B parameters) document parsing system. '
            'It uses PP-DocLayoutV2 for layout analysis and UniRec-0.1B for unified recognition of text, formulas, and tables.'
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type='pil', label='Input Image')
                btn = gr.Button('Analyze Document', variant='primary')
                download_output = gr.File(label='Download Results (MD, JSON)')

            with gr.Column():
                output_vis = gr.Image(type='pil', label='Layout Analysis')

        with gr.Tabs():
            with gr.TabItem('Markdown Preview'):
                output_md = gr.Markdown(label='Parsed Content')
            with gr.TabItem('Raw Markdown'):
                output_md_raw = gr.Textbox(label='Markdown Text', lines=20)
            with gr.TabItem('JSON Result'):
                output_json = gr.Code(label='JSON Result', language='json')

        btn.click(fn=process_image,
                  inputs=[input_img],
                  outputs=[
                      output_vis, output_md, output_json, download_output,
                      output_md_raw
                  ])

    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.launch(share=False)

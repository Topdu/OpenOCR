import os
import uuid
import shutil
import re
import base64
import gradio as gr
from PIL import Image

from tools.infer_doc import OpenDoc
from tools.utils.logging import get_logger

logger = get_logger(name='opendoc_gradio')

# Initialize the pipeline
pipeline: OpenDoc | None = None


def get_pipeline(gpu_id: int) -> OpenDoc:
    """获取或初始化OpenDoc流水线

    Args:
        gpu_id: GPU设备ID，-1表示使用CPU

    Returns:
        OpenDoc: 初始化好的OpenDoc实例
    """
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


def process_image(
    image_path: str | None
) -> tuple[Image.Image | None, str, str, str | None, str, str]:
    """处理图片并进行OCR识别

    Args:
        image_path: 图片文件路径，None表示无图片

    Returns:
        tuple: (可视化图片, Markdown内容(base64图片), JSON内容, ZIP文件路径, 原始Markdown, Markdown内容(base64图片))
    """
    if image_path is None:
        return None, '', '', None, '', ''

    # Get original image name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_ext = os.path.splitext(image_path)[1] or '.jpg'

    # Create a directory with image name for this request
    output_base_dir = 'gradio_outputs'
    os.makedirs(output_base_dir, exist_ok=True)

    # Add timestamp to avoid conflicts if same filename is uploaded multiple times
    timestamp = str(uuid.uuid4())[:8]
    folder_name = f'{base_name}_{timestamp}'
    tmp_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Copy and rename the input image
        tmp_img_path = os.path.join(tmp_dir, f'{base_name}{file_ext}')
        image = Image.open(image_path)
        image.save(tmp_img_path)

        # Predict
        output = list(
            current_pipeline.predict(tmp_img_path,
                                     use_doc_orientation_classify=False,
                                     use_doc_unwarping=False))
        if not output:
            return None, 'No results found.', '', None, '', ''

        res = output[0]

        # Save results
        res.save_to_img(tmp_dir)
        res.save_to_markdown(tmp_dir, pretty=True)
        res.save_to_json(tmp_dir)

        # Find the saved files
        vis_img = None
        for f in os.listdir(tmp_dir):
            if 'layout_order_res' in f:
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

        # Convert relative image paths to base64 for proper display in Gradio
        if markdown_content:

            def replace_img_with_base64(match):
                img_path = match.group(1)
                full_img_path = os.path.join(tmp_dir, img_path)

                if os.path.exists(full_img_path):
                    try:
                        with open(full_img_path, 'rb') as img_file:
                            img_data = base64.b64encode(
                                img_file.read()).decode('utf-8')
                            # Determine image format
                            ext = os.path.splitext(full_img_path)[1].lower()
                            mime_type = 'image/jpeg' if ext in [
                                '.jpg', '.jpeg'
                            ] else 'image/png'
                            # Replace src with base64 data URL
                            return match.group(0).replace(
                                f'src="{img_path}"',
                                f'src="data:{mime_type};base64,{img_data}"')
                    except Exception as e:
                        logger.warning(
                            f'Failed to convert image {img_path} to base64: {e}'
                        )
                return match.group(0)

            # Find all img tags and replace their src
            markdown_content_show = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>',
                                           replace_img_with_base64,
                                           markdown_content)
        else:
            markdown_content_show = markdown_content

        json_content = ''
        json_file_path = None
        for f in os.listdir(tmp_dir):
            if f.endswith('.json'):
                json_file_path = os.path.join(tmp_dir, f)
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    json_content = file.read()
                break

        # Prepare all files in tmp_dir for download by creating a zip archive
        zip_path = os.path.join(output_base_dir, f'{folder_name}.zip')
        _ = shutil.make_archive(zip_path.replace('.zip', ''), 'zip', tmp_dir)

        return vis_img, markdown_content_show, json_content, zip_path, markdown_content, markdown_content_show

    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        return None, f'Error during prediction: {str(e)}', '', None, '', ''


# Custom CSS with adaptive colors
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
#vis_output {
    min-height: 400px;
    border-radius: 12px;
    overflow: hidden;
}
#md_preview {
    max-height: 600px;
    min-height: 200px;
    overflow: auto;
    padding: 20px;
    background: var(--background-fill-primary);
    border-radius: 12px;
    box-shadow: var(--shadow-drop);
}
#md_preview img {
    display: block;
    margin: 16px auto;
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}
.notice {
    margin: 20px auto;
    max-width: 1200px;
    padding: 16px 20px;
    border-left: 4px solid var(--color-accent);
    border-radius: 8px;
    background: var(--background-fill-secondary);
    font-size: 14px;
    line-height: 1.8;
}
.notice strong {
    font-weight: 700;
    color: var(--color-accent);
}
.notice ul {
    margin-top: 8px;
    padding-left: 20px;
}
.notice li {
    margin: 8px 0;
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

# LaTeX delimiters for formula rendering
LATEX_DELIMS = [
    {
        'left': '$$',
        'right': '$$',
        'display': True
    },
    {
        'left': '$',
        'right': '$',
        'display': False
    },
    {
        'left': '\\(',
        'right': '\\)',
        'display': False
    },
    {
        'left': '\\[',
        'right': '\\]',
        'display': True
    },
]


# Define the Gradio Interface
def create_demo() -> gr.Blocks:
    """创建Gradio演示界面

    Returns:
        gr.Blocks: Gradio Blocks应用实例
    """
    with gr.Blocks(css=custom_css,
                   theme=gr.themes.Soft(),
                   title='OpenDoc-0.1B Demo') as demo:
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🚀 OpenDoc-0.1B</h1>
            <p>Ultra-Lightweight Document Parsing System with 0.1B Parameters (built by <a href="https://github.com/Topdu/OpenOCR">OCR Team</a>, <a href="https://fvl.fudan.edu.cn">FVL Lab</a>)</p>
            <p style="font-size: 0.95em; color: #888;">
                Powered by <a href="https://www.paddleocr.ai/latest/version3.x/module_usage/layout_analysis.html" target="_blank">PP-DocLayoutV2</a> for layout analysis and <a href="https://arxiv.org/pdf/2512.21095" target="_blank">UniRec-0.1B</a> for unified recognition of text, formulas, and tables
            </p>
        </div>
        <div class="quick-links">
            <a href="https://github.com/Topdu/OpenOCR" target="_blank">📖 GitHub</a>
            <a href="https://arxiv.org/pdf/2512.21095" target="_blank">📄 Paper</a>
            <a href="https://huggingface.co/topdu/unirec-0.1b" target="_blank">🤗 Model</a>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5, elem_classes=['upload-section']):
                input_img = gr.Image(type='filepath',
                                     label='📤 Upload Document Image',
                                     height=400)

                gr.Markdown("""
                ### 💡 Tips
                - Supports Chinese and English documents
                - Best for reports, papers, magazines, and complex layouts
                - Handles text, formulas, tables, and images
                """)

                btn = gr.Button('🔍 Analyze Document',
                                variant='primary',
                                size='lg')
                download_output = gr.File(label='📥 Download All Results (ZIP)',
                                          visible=True)

            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab('📝 Markdown Preview'):
                        output_md = gr.Markdown(
                            'Please upload an image and click "Analyze Document" to see results.',
                            latex_delimiters=LATEX_DELIMS,
                            elem_id='md_preview')
                    with gr.Tab('📊 Layout Visualization'):
                        output_vis = gr.Image(type='pil',
                                              label='Layout Analysis Results',
                                              elem_id='vis_output')

                    with gr.Tab('📄 Raw Markdown'):
                        output_md_raw = gr.Code(label='Markdown Source',
                                                language='markdown',
                                                lines=20)
                    with gr.Tab('📄 Raw Markdown with Base64 Images'):
                        output_md_raw_with_base64 = gr.Code(
                            label='Markdown Source',
                            language='markdown',
                            lines=20)

                    with gr.Tab('🗂️ JSON Result'):
                        output_json = gr.Code(label='Structured Data',
                                              language='json')

        # Feature notice
        gr.HTML("""
        <div class="notice">
            <strong>✨ Key Features:</strong>
            <ul>
                <li><strong>Ultra-lightweight:</strong> Only 0.1B parameters, fast inference speed</li>
                <li><strong>High accuracy:</strong> Achieves 90.57% on OmniDocBench (v1.5)</li>
                <li><strong>Unified recognition:</strong> Handles text, formulas, and tables in one model</li>
                <li><strong>Rich output:</strong> Provides Markdown, JSON, and visualization results</li>
            </ul>
        </div>
        """)

        btn.click(fn=process_image,
                  inputs=[input_img],
                  outputs=[
                      output_vis, output_md, output_json, download_output,
                      output_md_raw, output_md_raw_with_base64
                  ])

    return demo


if __name__ == '__main__':
    demo = create_demo()
    # Launch with larger interface settings
    demo.launch(share=False)

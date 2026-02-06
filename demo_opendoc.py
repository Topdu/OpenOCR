import os
import uuid
import shutil
import re
import base64
import argparse
import gradio as gr
from PIL import Image

from tools.infer_doc_onnx import OpenDocONNX
from tools.utils.logging import get_logger
from tools.download_example_images import get_example_images_path

logger = get_logger(name='opendoc_gradio')

# Initialize the pipeline
pipeline: OpenDocONNX | None = None


def get_pipeline(
    layout_model_path=None,
    unirec_encoder_path=None,
    unirec_decoder_path=None,
    tokenizer_mapping_path=None,
    use_gpu=None,
    auto_download=True
) -> OpenDocONNX:
    """获取或初始化OpenDocONNX流水线

    Args:
        layout_model_path: 版面检测ONNX模型路径
        unirec_encoder_path: UniRec编码器ONNX模型路径
        unirec_decoder_path: UniRec解码器ONNX模型路径
        tokenizer_mapping_path: Tokenizer映射文件路径
        use_gpu: Whether to use GPU. If None, auto-detect.
        auto_download: If True, automatically download missing models

    Returns:
        OpenDocONNX: 初始化好的OpenDocONNX实例
    """
    global pipeline
    if pipeline is None:
        gpu_info = 'GPU (auto-detect)' if use_gpu is None else ('GPU' if use_gpu else 'CPU')
        logger.info(f'Initializing OpenDoc ONNX pipeline on {gpu_info}...')
        pipeline = OpenDocONNX(
            layout_model_path=layout_model_path,
            unirec_encoder_path=unirec_encoder_path,
            unirec_decoder_path=unirec_decoder_path,
            tokenizer_mapping_path=tokenizer_mapping_path,
            use_gpu=use_gpu,
            auto_download=auto_download
        )
    return pipeline


# Ensure pipeline is initialized (will be done on first request)
current_pipeline = None


def process_image(
    image_path: str | None
) -> tuple[Image.Image | None, str, str, str | None, str, str]:
    """处理图片并进行OCR识别

    Args:
        image_path: 图片文件路径，None表示无图片

    Returns:
        tuple: (可视化图片, Markdown内容(base64图片), JSON内容, ZIP文件路径, 原始Markdown, Markdown内容(base64图片))
    """
    global current_pipeline

    if image_path is None:
        return None, '', '', None, '', ''

    # Initialize pipeline on first use
    if current_pipeline is None:
        current_pipeline = get_pipeline()

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
        result = current_pipeline(
            img_path=tmp_img_path,
            merge_layout_blocks=True
        )
        logger.info(f'Pipeline result type: {type(result)}, has content: {bool(result)}')
        if result:
            logger.info(f'Result keys: {result.keys()}')
            if 'recognition_results' in result:
                logger.info(f'Recognition results count: {len(result["recognition_results"])}')

        if not result:
            logger.warning('Pipeline returned empty result')
            return None, 'No results found.', '', None, '', ''

        # Save results
        logger.info(f'Saving results to: {tmp_dir}')
        current_pipeline.save_visualization(result, tmp_dir)
        logger.info('Visualization saved')
        current_pipeline.save_to_markdown(result, tmp_dir)
        logger.info('Markdown saved')
        current_pipeline.save_to_json(result, tmp_dir)
        logger.info('JSON saved')

        # The save methods create a subdirectory with the image name
        # Find the actual output directory
        actual_output_dir = None
        for item in os.listdir(tmp_dir):
            item_path = os.path.join(tmp_dir, item)
            if os.path.isdir(item_path):
                actual_output_dir = item_path
                break

        if actual_output_dir is None:
            # Fallback to tmp_dir if no subdirectory found
            actual_output_dir = tmp_dir

        logger.info(f'Actual output directory: {actual_output_dir}')
        logger.info(f'Files in output dir: {os.listdir(actual_output_dir)}')

        # Find the saved files
        vis_img = None
        for f in os.listdir(actual_output_dir):
            if f.endswith('_vis.jpg'):
                vis_img_path = os.path.join(actual_output_dir, f)
                vis_img = Image.open(vis_img_path)
                logger.info(f'Found visualization image: {vis_img_path}')
                break

        if vis_img is None:
            logger.warning('No visualization image found')

        markdown_content = ''
        md_file_path = None
        for f in os.listdir(actual_output_dir):
            if f.endswith('.md'):
                md_file_path = os.path.join(actual_output_dir, f)
                with open(md_file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                logger.info(f'Found markdown file: {md_file_path}, length: {len(markdown_content)}')
                break

        if not markdown_content:
            logger.warning('No markdown content found')

        # Convert relative image paths to base64 for proper display in Gradio
        if markdown_content:

            def replace_img_with_base64(match):
                img_path = match.group(1)
                full_img_path = os.path.join(actual_output_dir, img_path)

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
                                f'src=\"{img_path}\"',
                                f'src=\"data:{mime_type};base64,{img_data}\"')
                    except Exception as e:
                        logger.warning(
                            f'Failed to convert image {img_path} to base64: {e}')
                return match.group(0)

            # Find all img tags and replace their src
            markdown_content_show = re.sub(r'<img[^>]*src=\"([^\"]+)\"[^>]*>',
                                           replace_img_with_base64,
                                           markdown_content)
        else:
            markdown_content_show = markdown_content

        json_content = ''
        json_file_path = None
        for f in os.listdir(actual_output_dir):
            if f.endswith('.json'):
                json_file_path = os.path.join(actual_output_dir, f)
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    json_content = file.read()
                break
        # Prepare all files in tmp_dir for download by creating a zip archive
        zip_path = os.path.join(output_base_dir, f'{folder_name}.zip')
        _ = shutil.make_archive(zip_path.replace('.zip', ''), 'zip', tmp_dir)

        return vis_img, markdown_content_show, json_content, zip_path, markdown_content

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
    # Get example images path and download if necessary
    example_img_dir = get_example_images_path(demo_type='doc')

    # Get list of example images
    example_images = []
    if os.path.exists(example_img_dir):
        for file in os.listdir(example_img_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                example_images.append(os.path.join(example_img_dir, file))
        example_images = sorted(example_images)

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
            with gr.Column(scale=4, elem_classes=['upload-section']):
                input_img = gr.Image(type='filepath',
                                     label='📤 Upload Document Image',
                                     height=400)

                # Add examples if available
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=input_img,
                        label='📚 Example Documents'
                    )
                btn = gr.Button('🔍 Analyze Document',
                                variant='primary',
                                size='lg')
                gr.Markdown("""
                ### 💡 Tips
                - Supports Chinese and English documents
                - Best for reports, papers, magazines, and complex layouts
                - Handles text, formulas, tables, and images
                """)

                download_output = gr.File(label='📥 Download All Results (ZIP)',
                                          visible=True)

            with gr.Column(scale=6):
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
                      output_md_raw
                  ])

    return demo


def launch_demo(
    layout_model_path=None,
    unirec_encoder_path=None,
    unirec_decoder_path=None,
    tokenizer_mapping_path=None,
    use_gpu=None,
    auto_download=True,
    share=False,
    server_port=7860,
    server_name='0.0.0.0'
):
    """Launch OpenDoc ONNX Gradio demo with default configuration.

    Args:
        layout_model_path: Path to layout detection ONNX model (default: auto-download)
        unirec_encoder_path: Path to UniRec encoder ONNX model (default: auto-download)
        unirec_decoder_path: Path to UniRec decoder ONNX model (default: auto-download)
        tokenizer_mapping_path: Path to tokenizer mapping JSON (default: auto-download)
        use_gpu: Whether to use GPU. If None, auto-detect (default: None)
        auto_download: If True, automatically download missing models (default: True)
        share: Create a public share link (default: False)
        server_port: Server port (default: 7860)
        server_name: Server name (default: '0.0.0.0')

    Returns:
        gr.Blocks: Gradio demo instance
    """
    global current_pipeline

    # Initialize pipeline with arguments
    try:
        current_pipeline = get_pipeline(
            layout_model_path=layout_model_path,
            unirec_encoder_path=unirec_encoder_path,
            unirec_decoder_path=unirec_decoder_path,
            tokenizer_mapping_path=tokenizer_mapping_path,
            use_gpu=use_gpu,
            auto_download=auto_download
        )
    except Exception as e:
        logger.error(f'Failed to initialize pipeline: {e}')
        raise e

    demo = create_demo()
    # Launch with settings from arguments
    demo.launch(
        share=share,
        server_port=server_port,
        server_name=server_name
    )
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenDoc-0.1B ONNX Gradio Demo')

    # Model paths
    parser.add_argument('--layout-model',
                        type=str,
                        default=None,
                        help='Path to layout detection ONNX model (default: ~/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx)')
    parser.add_argument('--encoder',
                        type=str,
                        default=None,
                        help='Path to UniRec encoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx)')
    parser.add_argument('--decoder',
                        type=str,
                        default=None,
                        help='Path to UniRec decoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx)')
    parser.add_argument('--mapping',
                        type=str,
                        default=None,
                        help='Path to tokenizer mapping JSON (default: ~/.cache/openocrunirec_0_1b_onnx/unirec_tokenizer_mapping.json)')

    # GPU settings
    parser.add_argument('--use-gpu',
                        type=str,
                        default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Use GPU for inference (auto: auto-detect, true: force GPU, false: force CPU)')
    parser.add_argument('--no-auto-download',
                        action='store_true',
                        help='Disable automatic model download')

    # Gradio settings
    parser.add_argument('--share',
                        action='store_true',
                        help='Create a public link')
    parser.add_argument('--server-port',
                        type=int,
                        default=7860,
                        help='Server port')
    parser.add_argument('--server-name',
                        type=str,
                        default='0.0.0.0',
                        help='Server name')

    args = parser.parse_args()

    # Parse use_gpu argument
    if args.use_gpu == 'auto':
        use_gpu = None
    elif args.use_gpu == 'true':
        use_gpu = True
    else:
        use_gpu = False

    # Launch demo with parsed arguments
    launch_demo(
        layout_model_path=args.layout_model,
        unirec_encoder_path=args.encoder,
        unirec_decoder_path=args.decoder,
        tokenizer_mapping_path=args.mapping,
        use_gpu=use_gpu,
        auto_download=not args.no_auto_download,
        share=args.share,
        server_port=args.server_port,
        server_name=args.server_name
    )

import re
import gradio as gr
import numpy as np
from PIL import Image
from threading import Thread
import queue
import time

# Import ONNX inference components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.infer_unirec_onnx import UniRecONNX, clean_special_tokens
from tools.download_example_images import get_example_images_path
from tools.to_markdown import MarkdownConverter

# 创建全局 markdown_converter 实例
markdown_converter = MarkdownConverter()

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

# --- 1. Initialize ONNX Model ---
def initialize_model(
    encoder_path=None,
    decoder_path=None,
    mapping_path=None,
    use_gpu=None,
    auto_download=True
):
    """Initialize ONNX inference model.

    Args:
        encoder_path: Path to encoder ONNX model. If None, use default cache directory.
        decoder_path: Path to decoder ONNX model. If None, use default cache directory.
        mapping_path: Path to tokenizer mapping JSON. If None, use default cache directory.
        use_gpu: Whether to use GPU. If None, auto-detect. If True, force GPU. If False, force CPU.
        auto_download: If True, automatically download missing model files
    """
    print('Initializing UniRec ONNX model...')
    inference = UniRecONNX(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        mapping_path=mapping_path,
        use_gpu=use_gpu,
        auto_download=auto_download
    )
    print('✅ Model initialized successfully!')
    return inference


# Global model instance (will be initialized in main)
model = None


# --- 2. Streaming generation function ---
def stream_generate(inference, image, max_length=2048, result_queue=None):
    """Generate text with streaming output."""
    # Get token IDs
    bos_token_id = inference.tokenizer.bos_token_id
    eos_token_id = inference.tokenizer.eos_token_id
    pad_token_id = inference.tokenizer.pad_token_id

    # Encode image
    encoder_hidden_states, cross_k, cross_v = inference.encode_image(image)

    # Initialize generation
    generated_ids = [bos_token_id]

    # Initialize empty past_key_values
    batch_size = encoder_hidden_states.shape[0]
    past_key_values = []
    for _ in range(inference.num_decoder_layers):
        empty_key = np.zeros(
            (batch_size, inference.num_heads, 0, inference.head_dim),
            dtype=np.float32)
        empty_value = np.zeros(
            (batch_size, inference.num_heads, 0, inference.head_dim),
            dtype=np.float32)
        past_key_values.append((empty_key, empty_value))
    cleaned_text = ''
    put_token_num = 30
    # Generation loop with streaming
    for step in range(max_length - 1):
        current_token = generated_ids[-1]
        past_length = step

        # Decode step
        logits, past_key_values = inference.decode_step(
            current_token,
            past_length,
            cross_k,
            cross_v,
            past_key_values,
            padding_idx=pad_token_id
        )

        # Get next token
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token_id)

        # Decode current sequence and put in queue
        if result_queue is not None:
            current_text = inference.tokenizer.decode(generated_ids[-1:], skip_special_tokens=False)
            # print(current_text+'\n')
            cleaned_text = cleaned_text + clean_special_tokens(current_text)
            # Post-process HTML table attributes
            # cleaned_text = cleaned_text.replace('<tdcolspan=', '<td colspan=')
            # cleaned_text = cleaned_text.replace('<tdrowspan=', '<td rowspan=')
            # cleaned_text = cleaned_text.replace('"colspan=', '" colspan=')
            if (step + 1) % put_token_num == 0:
                result_queue.put(cleaned_text)
        result_queue.put(cleaned_text)
        # Check for EOS
        if next_token_id == eos_token_id:
            break

    # Signal completion
    if result_queue is not None:
        result_queue.put(None)


# --- 3. Gradio streaming function for dual display ---
def stream_recognize_image(input_image):
    """Stream recognition results with dual display: markdown text only during recognition, render after completion."""
    if input_image is None:
        yield '请先上传一张图片。', '**请先上传一张图片。**'
        return

    # Convert to PIL Image if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image).convert('RGB')
    else:
        input_image = input_image.convert('RGB')

    # Create queue for streaming results
    result_queue = queue.Queue()

    # Start generation in background thread
    thread = Thread(target=stream_generate, args=(model, input_image, 2048, result_queue))
    thread.daemon = True  # Set as daemon thread
    thread.start()

    # Stream results - only update markdown text, keep render area with "recognizing" message
    last_update_time = time.time()
    current_text = ''

    while True:
        try:
            # Get result with longer timeout
            result = result_queue.get(timeout=1.0)
            if result is None:  # Generation complete
                break
            current_text = result
            last_update_time = time.time()

            # Only update markdown text, show "recognizing" message in render area
            yield current_text, '_正在识别中，请稍候..._'

        except queue.Empty:
            # No new result yet, check if thread is still alive
            if not thread.is_alive():
                # Thread finished but no completion signal, break
                break
            # Yield current state periodically to keep UI responsive
            current_time = time.time()
            if current_time - last_update_time > 0.5:  # Update UI every 0.5s
                yield current_text if current_text else '正在识别中...', '_正在识别中，请稍候..._'
                last_update_time = current_time

    # Wait for thread to complete
    thread.join(timeout=2.0)

    # Final yield - now render the complete result
    formatted_result = format_markdown_output(current_text)
    yield formatted_result, formatted_result


def format_markdown_output(markdown_text):
    """Format markdown text for display.

    This function handles:
    - HTML tables (pass through as-is for Gradio Markdown rendering)
    - LaTeX formulas (already in proper format)
    - Basic markdown formatting
    """
    if not markdown_text:
        return '_等待识别结果..._'
    if '<table>' in markdown_text:
        markdown_text = markdown_converter._handle_table(markdown_text)
    if '\\(' in markdown_text or '\\[' in markdown_text:
        # extract the formula
        formula_pattern = r'\n\n\\\[.*?\\\]\n\n'
        # print(re.findall(formula_pattern, markdown_text, flags=re.DOTALL))
        # markdown_text = re.sub(formula_pattern, markdown_converter._handle_formula, markdown_text, flags=re.DOTALL)
        for formula in re.findall(formula_pattern, markdown_text, flags=re.DOTALL):
            markdown_text = markdown_text.replace(formula, markdown_converter._handle_formula(formula))
        if '\\(' in markdown_text:
            markdown_text = markdown_text.replace('\\(', '$')
            markdown_text = markdown_text.replace('\\)', '$')
    # Return the markdown text as-is
    # Gradio's Markdown component will handle the rendering
    return markdown_text


# --- 4. Gradio Interface ---
# Get example images path and download if necessary
example_img_dir = get_example_images_path(demo_type='unirec')

# Get list of example images
example_images = []
if os.path.exists(example_img_dir):
    for file in os.listdir(example_img_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            example_images.append(os.path.join(example_img_dir, file))
    example_images = sorted(example_images)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
            <h1 style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR">UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters</a></h1>
            <p style='text-align: center;'>0.1B超轻量模型统一文本与公式识别（由<a href="https://fvl.fudan.edu.cn">FVL实验室</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a> 创建）</p>
            <p style='text-align: center;'><a href="https://github.com/Topdu/OpenOCR/blob/main/docs/unirec.md">[本地GPU部署]</a>获取快速识别体验</p>"""
            )
    gr.Markdown('上传一张图片，点击"运行识别"按钮进行文本和公式识别。')
    with gr.Row():
        with gr.Column(scale=4):  # 左侧竖排：图片 + 按钮
            image_input = gr.Image(label='上传图片 or 粘贴截图', type='pil')

            # Add examples if available
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label='📚 示例图片'
                )

            with gr.Row():
                run_button = gr.Button('🚀 运行识别', variant='primary')
                clear_button = gr.Button('🗑️ 清空', variant='secondary')

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.Tab('📝 Markdown Source'):
                    markdown_output = gr.Code(label='Markdown Source',
                                            language='markdown',
                                            lines=20)
                with gr.Tab('📝 Markdown Preview'):
                    markdown_render = gr.Markdown(
                            value='_渲染后的表格/公式将显示在这里..._',
                            latex_delimiters=LATEX_DELIMS,
                            elem_id='md_preview')

    # 点击运行按钮后触发
    run_button.click(
        stream_recognize_image,
        inputs=[image_input],
        outputs=[markdown_output, markdown_render]
    )

    # 清空按钮功能：清空图片和输出结果
    def clear_all():
        return None, '', '_渲染后的表格/公式将显示在这里..._'

    clear_button.click(
        clear_all,
        outputs=[image_input, markdown_output, markdown_render]
    )


def launch_demo(
    encoder_path=None,
    decoder_path=None,
    mapping_path=None,
    use_gpu=None,
    auto_download=True,
    share=False,
    server_name='0.0.0.0',
    server_port=7860
):
    """Launch UniRec ONNX Gradio demo with default configuration.

    Args:
        encoder_path: Path to encoder ONNX model (default: auto-download)
        decoder_path: Path to decoder ONNX model (default: auto-download)
        mapping_path: Path to tokenizer mapping JSON (default: auto-download)
        use_gpu: Whether to use GPU. If None, auto-detect (default: None)
        auto_download: If True, automatically download missing models (default: True)
        share: Create a public share link (default: False)
        server_name: Server name for Gradio (default: '0.0.0.0')
        server_port: Server port for Gradio (default: 7860)

    Returns:
        gr.Blocks: Gradio demo instance
    """
    global model

    # Initialize model with specified parameters
    model = initialize_model(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        mapping_path=mapping_path,
        use_gpu=use_gpu,
        auto_download=auto_download
    )

    # Launch demo
    demo.queue().launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )
    return demo


# --- 5. Launch application ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UniRec ONNX Gradio Demo')
    parser.add_argument('--encoder_model',
                        type=str,
                        default=None,
                        help='Path to encoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx)')
    parser.add_argument('--decoder_model',
                        type=str,
                        default=None,
                        help='Path to decoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx)')
    parser.add_argument('--mapping',
                        type=str,
                        default=None,
                        help='Path to tokenizer mapping JSON (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json)')
    parser.add_argument('--use-gpu',
                        type=str,
                        default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Use GPU for inference (auto: auto-detect, true: force GPU, false: force CPU)')
    parser.add_argument('--no-auto-download',
                        action='store_true',
                        help='Disable automatic model download')
    parser.add_argument('--share',
                        action='store_true',
                        help='Create a public share link')
    parser.add_argument('--server-name',
                        type=str,
                        default='0.0.0.0',
                        help='Server name for Gradio')
    parser.add_argument('--server-port',
                        type=int,
                        default=7860,
                        help='Server port for Gradio')
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
        encoder_path=args.encoder_model,
        decoder_path=args.decoder_model,
        mapping_path=args.mapping,
        use_gpu=use_gpu,
        auto_download=not args.no_auto_download,
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )

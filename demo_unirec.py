import re
import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path

# Import ONNX inference components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.infer_unirec_onnx import UniRecONNX, clean_special_tokens
from tools.download_example_images import get_example_images_path
from tools.to_markdown import MarkdownConverter

# Create global markdown_converter instance
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
def stream_recognize_image(input_image, max_length=2048):
    """Stream recognition results with dual display: markdown text during recognition, render after completion.

    Args:
        input_image: Input PIL Image
        max_length: Maximum generation length
    """
    if input_image is None:
        yield 'Please upload an image first.', '**Please upload an image first.**'
        return

    # Convert to PIL Image if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image).convert('RGB')
    else:
        input_image = input_image.convert('RGB')

    # Get token IDs
    bos_token_id = model.tokenizer.bos_token_id
    eos_token_id = model.tokenizer.eos_token_id
    pad_token_id = model.tokenizer.pad_token_id

    # Encode image
    encoder_hidden_states, cross_k, cross_v = model.encode_image(input_image)

    # Initialize generation
    generated_ids = [bos_token_id]

    # Initialize empty past_key_values
    batch_size = encoder_hidden_states.shape[0]
    past_key_values = []
    for _ in range(model.num_decoder_layers):
        empty_key = np.zeros(
            (batch_size, model.num_heads, 0, model.head_dim),
            dtype=np.float32)
        empty_value = np.zeros(
            (batch_size, model.num_heads, 0, model.head_dim),
            dtype=np.float32)
        past_key_values.append((empty_key, empty_value))

    cleaned_text = ''

    # Generation loop with streaming
    for step in range(max_length - 1):
        current_token = generated_ids[-1]
        past_length = step

        # Decode step
        logits, past_key_values = model.decode_step(
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

        # Decode and clean current token
        current_text = model.tokenizer.decode(generated_ids[-1:], skip_special_tokens=False)
        cleaned_text += clean_special_tokens(current_text)

        yield cleaned_text, '_Recognizing, please wait..._'

        # Check for EOS
        if next_token_id == eos_token_id:
            break

    # Final yield - render the complete result
    if cleaned_text:
        formatted_result = format_markdown_output(cleaned_text)
        yield formatted_result, formatted_result
    else:
        yield 'Recognition failed, please try again.', '**Recognition failed, please try again.**'


def format_markdown_output(markdown_text):
    """Format markdown text for display.

    This function handles:
    - HTML tables (pass through as-is for Gradio Markdown rendering)
    - LaTeX formulas (already in proper format)
    - Basic markdown formatting
    """
    if not markdown_text:
        return '_Waiting for recognition results..._'
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

def create_demo() -> gr.Blocks:

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
.image-container img {
    max-width: 100%;
    max-height: 480px;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
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
            <h1><a href="https://github.com/Topdu/OpenOCR">UniRec-0.1B</a></h1>
            <p>Unified Text and Formula Recognition with 0.1B Parameters (built by <a href="https://fvl.fudan.edu.cn">FVL Lab</a> <a href="https://github.com/Topdu/OpenOCR">OCR Team</a>)</p>
        </div>
        <div class="quick-links">
            <a href="https://github.com/Topdu/OpenOCR" target="_blank">📖 GitHub</a>
            <a href="https://arxiv.org/pdf/2512.21095" target="_blank">📄 Paper</a>
            <a href="https://huggingface.co/topdu/unirec-0.1b" target="_blank">🤗 Model</a>
            <a href="https://github.com/Topdu/OpenOCR/blob/main/docs/unirec.md" target="_blank">🚀 Local GPU Deployment</a>
        </div>
        """)
        gr.Markdown('Upload an image and click the "Run Recognition" button to recognize text and formulas.')
        with gr.Row():
            with gr.Column(scale=4, elem_classes=['upload-section']):  # Left column: image + buttons
                image_input = gr.Image(label='Upload Image or Paste Screenshot', type='pil', elem_classes=['image-container'])

                # Add examples if available
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                    label='📚 Example Images'
                    )

                with gr.Row():
                    run_button = gr.Button('🚀 Run Recognition', variant='primary')
                    clear_button = gr.Button('🗑️ Clear', variant='secondary')

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab('📄 Markdown Source'):
                        markdown_output = gr.Code(label='Markdown Source',
                                                language='markdown',
                                                lines=20)
                    with gr.Tab('📝 Markdown Preview'):
                        markdown_render = gr.Markdown(
                                value='_Rendered tables/formulas will be displayed here..._',
                                latex_delimiters=LATEX_DELIMS,
                                elem_id='md_preview')

        # Trigger recognition on button click
        run_button.click(
            stream_recognize_image,
            inputs=[image_input],
            outputs=[markdown_output, markdown_render]
        )

        # Clear button: reset image and output
        def clear_all():
            return None, '', '_Rendered tables/formulas will be displayed here..._'

        clear_button.click(
            clear_all,
            outputs=[image_input, markdown_output, markdown_render]
        )
    return demo


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
    demo = create_demo()

    allowed_paths = str(Path.home() / '.cache' / 'openocr')
    # Launch demo
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        allowed_paths=[allowed_paths]
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

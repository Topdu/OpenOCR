"""
ONNX inference script for UniRec model.
Standalone version without transformers dependency.

Version: Optimized v2
- Supports optimized KV cache format: [batch_size, num_heads, seq_len, head_dim]
- Compatible with merged QKV/KV projection models
- No reshape overhead during generation
"""

import json
import os
import re
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image


def download_model_files(model_dir=None):
    """Download ONNX model files from ModelScope or HuggingFace.

    Args:
        model_dir: Directory to save model files. If None, use default cache directory.

    Returns:
        Tuple of (encoder_path, decoder_path, mapping_path)
    """
    # Use default cache directory if not specified
    if model_dir is None:
        cache_dir = Path.home() / '.cache' / 'openocr'
        model_dir = cache_dir / 'unirec_0_1b_onnx'
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        'unirec_encoder.onnx',
        'unirec_decoder.onnx',
        'unirec_tokenizer_mapping.json'
    ]

    # Check which files are missing
    missing_files = [f for f in required_files if not (model_dir / f).exists()]

    if not missing_files:
        print(f'✅ All model files found in {model_dir}')
        return tuple(str(model_dir / f) for f in required_files)

    print(f'📥 Missing files: {missing_files}')
    print(f'📥 Downloading model files to {model_dir}...')

    download_success = False

    try:
        # Try ModelScope first (default)
        print('🌐 Trying ModelScope (China mirror) first...')
        try:
            from modelscope import snapshot_download
            model_path = snapshot_download(
                'topdktu/unirec_0_1b_onnx',
                cache_dir=str(model_dir.parent)
            )
            print(f'✅ Downloaded to {model_path}')

            # Copy files to target directory
            import shutil
            for file in required_files:
                src = Path(model_path) / file
                dst = model_dir / file
                if src.exists() and not dst.exists():
                    shutil.copy(str(src), str(dst))
                    print(f'  ✓ {file}')

            # Verify all files exist after download
            all_files_exist = all((model_dir / f).exists() for f in required_files)
            if all_files_exist:
                download_success = True
                print('✅ All files downloaded successfully from ModelScope!')
            else:
                print('⚠️  ModelScope download incomplete, trying HuggingFace...')

        except ImportError:
            print('⚠️  modelscope not installed. Install with: pip install modelscope')
            print('   Trying HuggingFace...')
        except Exception as e:
            print(f'⚠️  ModelScope download failed: {e}')
            print('   Trying HuggingFace...')

        if not download_success:
            # Try HuggingFace
            print('🌐 Using HuggingFace...')
            try:
                from huggingface_hub import hf_hub_download

                for file in missing_files:
                    print(f'  Downloading {file}...')
                    downloaded_path = hf_hub_download(
                        repo_id='topdu/unirec_0_1b_onnx',
                        filename=file,
                        cache_dir=str(model_dir.parent),
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False
                    )
                    print(f'  ✓ {file}')

                # Verify all files exist after download
                all_files_exist = all((model_dir / f).exists() for f in required_files)
                if all_files_exist:
                    download_success = True
                    print('✅ All files downloaded successfully from HuggingFace!')

            except ImportError:
                print('⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub')
                raise RuntimeError(
                    'Cannot download models. Please install either:\n'
                    '  - huggingface_hub: pip install huggingface_hub\n'
                    '  - modelscope: pip install modelscope\n'
                    'Or manually download from:\n'
                    '  - https://huggingface.co/topdu/unirec_0_1b_onnx\n'
                    '  - https://modelscope.cn/models/topdktu/unirec_0_1b_onnx'
                )

        if not download_success:
            raise RuntimeError(
                'Failed to download all required files. Please manually download from:\n'
                '  - https://huggingface.co/topdu/unirec_0_1b_onnx\n'
                '  - https://modelscope.cn/models/topdktu/unirec_0_1b_onnx'
            )
    except Exception as e:
        print(f'❌ Download failed: {e}')
        print('\n📝 Manual download instructions:')
        print('   1. Visit: https://huggingface.co/topdu/unirec_0_1b_onnx')
        print('      or: https://modelscope.cn/models/topdktu/unirec_0_1b_onnx')
        print(f'   2. Download these files to {model_dir}:')
        for file in required_files:
            print(f'      - {file}')
        raise

    return tuple(str(model_dir / f) for f in required_files)


def check_and_download_models(encoder_path, decoder_path, mapping_path, auto_download=True):
    """Check if model files exist, download if missing.

    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model
        mapping_path: Path to tokenizer mapping JSON
        auto_download: If True, automatically download missing files

    Returns:
        Tuple of (encoder_path, decoder_path, mapping_path) with verified paths
    """
    files_to_check = {
        'encoder': encoder_path,
        'decoder': decoder_path,
        'mapping': mapping_path
    }

    missing_files = {k: v for k, v in files_to_check.items() if not os.path.exists(v)}

    if not missing_files:
        return encoder_path, decoder_path, mapping_path

    print('⚠️  Missing model files:')
    for name, path in missing_files.items():
        print(f'   - {name}: {path}')

    if not auto_download:
        raise FileNotFoundError(
            'Model files not found. Please download from:\n'
            '  - https://huggingface.co/topdu/unirec_0_1b_onnx\n'
            '  - https://modelscope.cn/models/topdktu/unirec_0_1b_onnx'
        )

    # Determine model directory from encoder path
    encoder_dir = os.path.dirname(encoder_path)
    if encoder_dir and encoder_dir != './unirec_0_1b_onnx':
        # User specified a custom path
        model_dir = encoder_dir
    else:
        # Use default cache directory
        model_dir = None

    # Try ModelScope first (faster in China), then HuggingFace
    try:
        print('🇨🇳 Trying ModelScope (China mirror) first...')
        return download_model_files(model_dir)
    except:
        print('🌍 Trying HuggingFace...')
        return download_model_files(model_dir)


class SimpleImageProcessor:
    """Standalone image processor without transformers dependency."""

    def __init__(
            self,
            max_side=(960, 1408),  # (width, height)
            divided_factor=(64, 64),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
    ):
        self.max_side = max_side
        self.divided_factor = divided_factor
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def _calculate_target_size(self, original_width, original_height):
        """Calculate target size with aspect ratio preservation."""
        max_width, max_height = self.max_side
        aspect_ratio = original_width / original_height

        if original_width > max_width or original_height > max_height:
            if (max_width / max_height) >= aspect_ratio:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
        else:
            new_width, new_height = original_width, original_height

        # Apply divided factor
        div_w, div_h = self.divided_factor
        final_width = max(int(new_width // div_w * div_w), 64)
        final_height = max(int(new_height // div_h * div_h), 64)

        return (final_width, final_height)

    def __call__(self, image):
        """
        Process image for model input.

        Args:
            image: PIL Image

        Returns:
            dict with 'pixel_values' as numpy array [1, 3, H, W]
        """
        if not isinstance(image, Image.Image):
            raise ValueError('Input must be PIL Image')

        original_width, original_height = image.size

        # Resize
        target_size = self._calculate_target_size(original_width,
                                                  original_height)
        image = image.resize(target_size, resample=Image.BICUBIC)

        # Convert to numpy array [H, W, C] and normalize to [0, 1]
        image_np = np.array(image, dtype=np.float32)[:, :, :3] / 255.0

        # Normalize: (x - mean) / std
        image_np = (image_np - self.image_mean) / self.image_std

        # Transpose to [C, H, W]
        image_np = image_np.transpose(2, 0, 1)

        # Add batch dimension [1, C, H, W]
        image_np = np.expand_dims(image_np, axis=0)

        return {'pixel_values': image_np}


class SimpleTokenizer:
    """Standalone tokenizer without transformers dependency."""

    def __init__(self, mapping_file=None):
        """
        Load vocabulary from mapping file or tokenizer.json.

        Args:
            vocab_file: path to tokenizer.json (deprecated, use mapping_file)
            mapping_file: path to unirec_tokenizer_mapping.json (recommended)
        """

        if mapping_file and os.path.exists(mapping_file):
            # 使用导出的映射文件 (推荐)
            print(f'Loading tokenizer from mapping file: {mapping_file}')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)

            # 直接使用 id_to_token 映射
            self.id_to_token = {
                int(k): v
                for k, v in mapping_data['id_to_token'].items()
            }
            self.vocab_size = mapping_data['vocab_size']

            # 特殊 token
            special_tokens = mapping_data['special_tokens']
            self.bos_token_id = special_tokens['bos_token_id']
            self.eos_token_id = special_tokens['eos_token_id']
            self.pad_token_id = special_tokens['pad_token_id']

        print(f'✅ Loaded vocabulary with {self.vocab_size} tokens')

    def decode(self, token_ids, skip_special_tokens=False):
        """
        Decode token IDs to text.

        Args:
            token_ids: list of token IDs
            skip_special_tokens: whether to skip special tokens

        Returns:
            decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token_id in [
                        self.bos_token_id, self.eos_token_id, self.pad_token_id
                ]:
                    continue

                tokens.append(token)
            else:
                tokens.append(f'<unk_{token_id}>')

        # Join tokens
        text = ''.join(tokens)

        return text


def clean_special_tokens(text):
    """Clean special tokens from decoded text."""
    # Remove special formatting tokens
    text = text.replace('Ġ', ' ').replace('Ċ', '\n')
    text = text.replace('<|bos|>', '').replace('<|eos|>',
                                               '').replace('<|pad|>', '')

    # Apply regex rules
    rules = [
        (r'-<\|sn\|>', ''),
        (r' <\|sn\|>', ' '),
        (r'<\|sn\|>', ' '),
        (r'<\|unk\|>', ''),
        (r'<s>', ''),
        (r'</s>', ''),
        (r'\uffff', ''),
        (r'_{4,}', '___'),
        (r'\.{4,}', '...'),
    ]

    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)

    return text


class UniRecONNX:
    """ONNX-based inference for UniRec model (standalone version)."""

    def __init__(
        self,
        encoder_path=None,
        decoder_path=None,
        mapping_path=None,
        use_gpu=None,
        auto_download=True,
    ):
        """Initialize ONNX inference sessions.

        Args:
            encoder_path: Path to encoder ONNX model. If None, use default cache directory.
            decoder_path: Path to decoder ONNX model. If None, use default cache directory.
            mapping_path: Path to tokenizer mapping JSON. If None, use default cache directory.
            use_gpu: Whether to use GPU. If None, auto-detect. If True, force GPU. If False, force CPU.
            auto_download: If True, automatically download missing model files
        """
        # Set default paths if not provided
        if encoder_path is None or decoder_path is None or mapping_path is None:
            cache_dir = Path.home() / '.cache' / 'openocr'
            model_path = cache_dir / 'unirec_0_1b_onnx'
            if encoder_path is None:
                encoder_path = str(model_path / 'unirec_encoder.onnx')
            if decoder_path is None:
                decoder_path = str(model_path / 'unirec_decoder.onnx')
            if mapping_path is None:
                mapping_path = str(model_path / 'unirec_tokenizer_mapping.json')

        # Check and download models if needed
        encoder_path, decoder_path, mapping_path = check_and_download_models(
            encoder_path, decoder_path, mapping_path, auto_download=auto_download
        )

        print('Loading ONNX models...')

        # Determine execution provider
        providers = self._get_execution_providers(use_gpu)
        print(f'Using execution providers: {providers}')

        # Create ONNX runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)

        # Initialize processor and tokenizer
        self.processor = SimpleImageProcessor()
        self.tokenizer = SimpleTokenizer(mapping_file=mapping_path)

        # Get model info from decoder session
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        self.num_decoder_layers = None
        self.num_heads = None
        self.head_dim = None

        for inp in self.decoder_session.get_inputs():
            if 'past_key' in inp.name:
                layer_idx = int(inp.name.split('_')[-1])
                if self.num_decoder_layers is None or layer_idx + 1 > self.num_decoder_layers:
                    self.num_decoder_layers = layer_idx + 1
                # Get shape info: [batch_size, num_heads, seq_len, head_dim]
                if len(inp.shape) == 4:
                    if self.num_heads is None and isinstance(
                            inp.shape[1], int):
                        self.num_heads = inp.shape[1]
                    if self.head_dim is None and isinstance(inp.shape[3], int):
                        self.head_dim = inp.shape[3]

        # Calculate d_model
        if self.num_heads and self.head_dim:
            self.d_model = self.num_heads * self.head_dim
        else:
            self.d_model = None

        print('\n✅ Models loaded successfully!')
        print(f'   Number of decoder layers: {self.num_decoder_layers}')
        print(f'   Number of attention heads: {self.num_heads}')
        print(f'   Head dimension: {self.head_dim}')
        print(f'   Model dimension (d_model): {self.d_model}')
        print(f'   Vocabulary size: {self.tokenizer.vocab_size}')

    def _get_execution_providers(self, use_gpu):
        """Determine execution providers based on GPU availability and user preference.

        Args:
            use_gpu: None (auto-detect), True (force GPU), or False (force CPU)

        Returns:
            List of execution providers in priority order
        """
        available_providers = ort.get_available_providers()

        if use_gpu is False:
            # Force CPU
            print('🔧 User specified: Using CPU')
            return ['CPUExecutionProvider']

        # Check for GPU providers
        gpu_providers = []
        if 'CUDAExecutionProvider' in available_providers:
            gpu_providers.append('CUDAExecutionProvider')
        # if 'TensorrtExecutionProvider' in available_providers:
        #     gpu_providers.append('TensorrtExecutionProvider')

        if use_gpu is True:
            # Force GPU
            if gpu_providers:
                print(f'🔧 User specified: Using GPU ({gpu_providers[0]})')
                return gpu_providers + ['CPUExecutionProvider']
            else:
                print('⚠️  GPU requested but not available, falling back to CPU')
                return ['CPUExecutionProvider']

        # Auto-detect (use_gpu is None)
        if gpu_providers:
            print(f'✅ GPU detected: Using {gpu_providers[0]}')
            return gpu_providers + ['CPUExecutionProvider']
        else:
            print('ℹ️  No GPU detected, using CPU')
            return ['CPUExecutionProvider']

    def encode_image(self, image):
        """Encode image using encoder ONNX model."""
        # Preprocess image
        data_img = self.processor(image)
        pixel_values = data_img['pixel_values']

        # Run encoder
        encoder_outputs = self.encoder_session.run(
            None, {'pixel_values': pixel_values.astype(np.float32)})

        encoder_hidden_states = encoder_outputs[0]
        cross_k = encoder_outputs[1]
        cross_v = encoder_outputs[2]

        return encoder_hidden_states, cross_k, cross_v

    def decode_step(self,
                    input_id,
                    past_length,
                    cross_k,
                    cross_v,
                    past_key_values,
                    padding_idx=1):
        """Unified decoder step with or without cache."""
        # Prepare inputs
        input_ids = np.array([[input_id]], dtype=np.int64)
        # Use M2M100's position ID calculation with past_key_values_length
        position_ids = np.array([[padding_idx + 1 + past_length]],
                                dtype=np.int64)

        decoder_inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'cross_k': cross_k.astype(np.float32),
            'cross_v': cross_v.astype(np.float32),
        }

        # Add past_key_values
        for i, (past_key, past_value) in enumerate(past_key_values):
            decoder_inputs[f'past_key_{i}'] = past_key.astype(np.float32)
            decoder_inputs[f'past_value_{i}'] = past_value.astype(np.float32)

        # Run decoder
        decoder_outputs = self.decoder_session.run(None, decoder_inputs)

        # Parse outputs
        logits = decoder_outputs[0]

        # Extract present_key_values
        present_key_values = []
        for i in range(self.num_decoder_layers):
            key = decoder_outputs[1 + i * 2]
            value = decoder_outputs[1 + i * 2 + 1]
            present_key_values.append((key, value))

        return logits, present_key_values

    def __call__(
        self,
        img_path=None,
        img_numpy=None,
        image=None,
        max_length=2048,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        Unified interface for UniRec inference.

        Args:
            img_path: Path to input image (str or Path)
            img_numpy: Input image as numpy array (BGR format)
            image: PIL Image object (RGB format)
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Tuple of (generated_text, generated_ids)
        """
        # Load image from path, numpy array, or use provided PIL image
        if img_path is not None:
            image = Image.open(img_path).convert('RGB')
        elif img_numpy is not None:
            # Convert BGR to RGB if needed
            if len(img_numpy.shape) == 3 and img_numpy.shape[2] == 3:
                import cv2
                img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_numpy)
        elif image is None:
            raise ValueError('Either img_path, img_numpy, or image must be provided')

        # Get token IDs
        if bos_token_id is None:
            bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        # Encode image
        print('Encoding image...')
        t_start = time.time()
        encoder_hidden_states, cross_k, cross_v = self.encode_image(image)
        print(f'Encoding time: {time.time() - t_start:.2f} seconds')
        print(f'  cross_k shape: {cross_k.shape}')
        print(f'  cross_v shape: {cross_v.shape}')

        # Initialize generation
        print('Generating text...')
        generated_ids = [bos_token_id]

        # Initialize empty past_key_values for first step
        # Shape: [batch_size, num_heads, 0, head_dim]
        batch_size = encoder_hidden_states.shape[0]
        past_key_values = []
        for _ in range(self.num_decoder_layers):
            empty_key = np.zeros(
                (batch_size, self.num_heads, 0, self.head_dim),
                dtype=np.float32)
            empty_value = np.zeros(
                (batch_size, self.num_heads, 0, self.head_dim),
                dtype=np.float32)
            past_key_values.append((empty_key, empty_value))

        # Generation loop
        t_start = time.time()
        for step in range(max_length - 1):
            # Current token to decode
            current_token = generated_ids[-1]

            # past_length is the sequence length in cache
            past_length = step

            # Decode step
            logits, past_key_values = self.decode_step(
                current_token,
                past_length,
                cross_k,
                cross_v,
                past_key_values,
                padding_idx=pad_token_id)

            # Get next token
            next_token_id = int(np.argmax(logits[0, -1, :]))
            generated_ids.append(next_token_id)

            # Check for EOS
            if next_token_id == eos_token_id:
                break

            # Progress indicator
            if (step + 1) % 50 == 0:
                print(f'  Generated {step + 1} tokens...')

        t_end = time.time()
        print(f'✅ Generation complete! Total tokens: {len(generated_ids)}')
        print(f'  Time taken: {t_end - t_start:.2f} seconds')
        print(
            f'  Tokens per second: {len(generated_ids) / (t_end - t_start):.2f}'
        )

        # Decode tokens
        generated_text = self.tokenizer.decode(generated_ids,
                                               skip_special_tokens=False)
        cleaned_text = clean_special_tokens(generated_text)

        return cleaned_text, generated_ids


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='UniRec ONNX Inference (Standalone)')
    parser.add_argument('--image',
                        type=str,
                        required=True,
                        help='Path to input image')
    parser.add_argument('--encoder-model',
                        type=str,
                        default=None,
                        help='Path to encoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx)')
    parser.add_argument('--decoder-model',
                        type=str,
                        default=None,
                        help='Path to decoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx)')
    parser.add_argument(
        '--mapping',
        type=str,
        default=None,
        help='Path to tokenizer mapping JSON (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json)')
    parser.add_argument('--max-length',
                        type=int,
                        default=2048,
                        help='Maximum generation length')
    parser.add_argument('--use-gpu',
                        type=str,
                        default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Use GPU for inference (auto: auto-detect, true: force GPU, false: force CPU)')
    parser.add_argument('--no-auto-download',
                        action='store_true',
                        help='Disable automatic model download')
    args = parser.parse_args()

    # Parse use_gpu argument
    if args.use_gpu == 'auto':
        use_gpu = None
    elif args.use_gpu == 'true':
        use_gpu = True
    else:
        use_gpu = False

    # Load image
    print(f'Loading image: {args.image}')
    image = Image.open(args.image).convert('RGB')

    # Initialize inference
    inference = UniRecONNX(
        encoder_path=args.encoder_model,
        decoder_path=args.decoder_model,
        mapping_path=args.mapping,
        use_gpu=use_gpu,
        auto_download=not args.no_auto_download,
    )

    # Generate
    result_text, generated_ids = inference(
        image=image,
        max_length=args.max_length,
    )

    # Print result
    print('\n' + '=' * 80)
    print('RESULT:')
    print('=' * 80)
    print(result_text)
    print('=' * 80)
    print(f'\nGenerated {len(generated_ids)} tokens')


if __name__ == '__main__':
    main()

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
import numpy as np
import onnxruntime as ort
from PIL import Image


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


class UniRecONNXInference:
    """ONNX-based inference for UniRec model (standalone version)."""

    def __init__(
        self,
        encoder_path='./unirec_0_1b_onnx/unirec_encoder.onnx',
        decoder_path='./unirec_0_1b_onnx/unirec_decoder.onnx',
        mapping_path='./unirec_0_1b_onnx/unirec_tokenizer_mapping.json',
    ):
        """Initialize ONNX inference sessions."""
        print('Loading ONNX models...')

        # Create ONNX runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options)
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options)

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

    def generate(
        self,
        image,
        max_length=2048,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """Generate text from image."""
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
    parser.add_argument('--encoder',
                        type=str,
                        default='./unirec_0_1b_onnx/unirec_encoder.onnx',
                        help='Path to encoder ONNX model')
    parser.add_argument('--decoder',
                        type=str,
                        default='./unirec_0_1b_onnx/unirec_decoder.onnx',
                        help='Path to decoder ONNX model')
    parser.add_argument(
        '--mapping',
        type=str,
        default='./unirec_0_1b_onnx/unirec_tokenizer_mapping.json',
        help='Path to tokenizer mapping JSON')
    parser.add_argument('--max-length',
                        type=int,
                        default=2048,
                        help='Maximum generation length')
    args = parser.parse_args()

    # Load image
    print(f'Loading image: {args.image}')
    image = Image.open(args.image).convert('RGB')

    # Initialize inference
    inference = UniRecONNXInference(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        mapping_path=args.mapping,
    )

    # Generate
    result_text, generated_ids = inference.generate(
        image,
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

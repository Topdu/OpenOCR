from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path

import os
import json
import time
import argparse
from typing import Dict, Optional, Union, List, Tuple

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from utils.logging import get_logger
from utils.utility import get_image_file_list

from utils.opendoc_onnx_utils.utils import (
    convert_otsl_to_html,
    crop_margin,
    filter_overlap_boxes,
    merge_blocks,
    tokenize_figure_of_table,
    truncate_repetitive_content,
    untokenize_figure_of_table,
)
from to_markdown import MarkdownConverter
from depolyment.unirec_onnx.infer_onnx import (
    SimpleImageProcessor,
    SimpleTokenizer,
    clean_special_tokens,
)

# 创建全局 markdown_converter 实例
markdown_converter = MarkdownConverter()

logger = get_logger(name='opendoc_onnx')

root_dir = Path(__file__).resolve().parent

IMAGE_LABELS = ['image', 'header_image', 'footer_image', 'seal']


def _get_image_name_and_dir(result: Dict, output_path: str):
    """根据图片名创建子目录并返回(img_name, img_dir)"""
    img_name = os.path.basename(result['input_path'])
    if '.' in img_name:
        img_name = img_name.rsplit('.', 1)[0]

    img_dir = os.path.join(output_path, img_name)
    os.makedirs(img_dir, exist_ok=True)

    return img_name, img_dir


# ==================== Layout Detection ONNX ====================
class LayoutDetectorONNX:
    """ONNX版本的版面检测模型"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 threshold: float = 0.5):
        """
        初始化ONNX版面检测模型

        Args:
            model_path: ONNX模型路径
            device: 'cpu' 或 'cuda'
            threshold: 检测阈值
        """
        self.threshold = threshold
        logger.info(f"Loading layout detection ONNX model from {model_path}")

        # 设置ONNX Runtime的执行提供者
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # 创建ONNX Runtime会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path,
                                            sess_options,
                                            providers=providers)

        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

        logger.info(f"✅ Layout model loaded successfully on {device}")
        logger.info(f"   Input names: {self.input_names}")
        logger.info(f"   Output names: {self.output_names}")

        self.label_map = {
            0: 'abstract',
            1: 'algorithm',
            2: 'aside_text',
            3: 'chart',
            4: 'content',
            5: 'display_formula',
            6: 'doc_title',
            7: 'figure_title',
            8: 'footer',
            9: 'footer_image',
            10: 'footnote',
            11: 'formula_number',
            12: 'header',
            13: 'header_image',
            14: 'image',
            15: 'inline_formula',
            16: 'number',
            17: 'paragraph_title',
            18: 'reference',
            19: 'reference_content',
            20: 'seal',
            21: 'table',
            22: 'text',
            23: 'vertical_text',
            24: 'vision_footnote'
        }

    def crop_by_boxes(self, image: np.ndarray,
                      boxes: List[Dict]) -> List[Dict]:
        """
        根据检测框裁剪图像区域

        Args:
            image: BGR格式的原始图像
            boxes: 检测框列表

        Returns:
            包含裁剪图像的块列表
        """
        blocks = []
        for box in boxes:
            coord = box['coordinate']
            x1, y1, x2, y2 = map(int, coord)

            # 裁剪图像
            cropped_img = image[y1:y2, x1:x2]
            if cropped_img.size == 0:
                cropped_img = None

            blocks.append({
                'img': cropped_img,
                'box': coord,
                'label': box['label'],
                'score': box.get('score', 1.0),
                'cls_id': box.get('cls_id', -1),
                'custom_value': box.get('custom_value', 0),
            })
        return blocks

    def preprocess(
        self, image: np.ndarray, target_input_size: tuple = (800, 800)
    ) -> Tuple[Dict, Tuple[float, float], int, int]:
        """
        Args:
            image: BGR格式的图像
            target_input_size: 目标尺寸 (height, width)

        Returns:
            输入字典, (scale_h, scale_w), 原始高度, 原始宽度
        """
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]

        #  Resize (keep_ratio=false, interp=2)
        target_h, target_w = target_input_size
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        new_h, new_w = int(orig_h * scale_h), int(orig_w * scale_w)
        resized = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_blob = resized_rgb.astype(np.float32) / 255.0

        input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]

        preprocess_shape = np.array([[target_h, target_w]], dtype=np.float32)

        # scale_factor: [[scale_h, scale_w]]
        scale_factor = np.array([[scale_h, scale_w]], dtype=np.float32)

        inputs = {
            'im_shape': preprocess_shape,  # shape: [1, 2]
            'image': input_blob.astype(np.float32),
            'scale_factor': scale_factor  # shape: [1, 2]
        }

        return inputs, (scale_h, scale_w), orig_h, orig_w

    def postprocess(
        self,
        image: np.ndarray,
        outputs: list,
        scale: Tuple[float, float],
        ori_h: int,
        ori_w: int,
        merge_layout_blocks: bool = True,
        use_chart_recognition: bool = False,
    ) -> Dict:
        """
        后处理，仿照 get_layout_parsing_results 的逻辑

        Args:
            image: 原始图像 (BGR格式)
            outputs: 模型输出
            scale: 缩放因子 (scale_h, scale_w)
            ori_h: 原始高度
            ori_w: 原始宽度
            merge_layout_blocks: 是否合并布局块
            use_chart_recognition: 是否识别图表

        Returns:
            检测结果字典，包含 boxes 和 blocks
        """
        # PaddleDetection ONNX 输出格式:
        # outputs[0]: bbox [N, 8] - 前6个值: [class_id, score, x1, y1, x2, y2]
        bboxes = outputs[0]  # [N, 8]

        # 如果没有检测到任何框
        if bboxes.shape[0] == 0:
            return {'boxes': [], 'blocks': []}

        # 过滤低置信度的框
        filtered_bboxes = bboxes[bboxes[:, 1] > self.threshold]

        if filtered_bboxes.shape[0] == 0:
            return {'boxes': [], 'blocks': []}

        # 解析每个检测框
        result_boxes = []
        for bbox in filtered_bboxes:
            class_id = int(bbox[0])
            score = float(bbox[1])
            order_value = float(bbox[6])
            x1, y1, x2, y2 = bbox[2:6]

            # 裁剪到图像边界
            x1 = float(np.clip(x1, 0, ori_w))
            y1 = float(np.clip(y1, 0, ori_h))
            x2 = float(np.clip(x2, 0, ori_w))
            y2 = float(np.clip(y2, 0, ori_h))

            result_boxes.append({
                'cls_id':
                class_id,
                'label':
                self.label_map.get(class_id, f'class_{class_id}'),
                'score':
                score,
                'coordinate': [x1, y1, x2, y2],
                'custom_value':
                order_value
            })

        result_dict = {'boxes': result_boxes}

        # 去除重叠框
        result_dict = filter_overlap_boxes(result_dict)

        # 根据 custom_value 排序
        result_dict['boxes'] = sorted(result_dict['boxes'],
                                      key=lambda box: box['custom_value'],
                                      reverse=False)

        # 给每个 label 添加顺序编号
        for idx, box in enumerate(result_dict['boxes'], start=1):
            base_label = box['label']
            box['label'] = f"{base_label}_{idx: 02d}"

        # 裁剪图像区域
        blocks = self.crop_by_boxes(image, result_dict['boxes'])

        # 确定 image_labels
        image_labels = IMAGE_LABELS if use_chart_recognition else IMAGE_LABELS + [
            'chart'
        ]

        # 合并布局块
        if merge_layout_blocks:
            blocks = merge_blocks(blocks,
                                  non_merge_labels=image_labels + ['table'])

        result_dict['blocks'] = blocks

        return result_dict

    def __call__(self,
                 images: Union[np.ndarray, List[np.ndarray]],
                 threshold: Optional[float] = None) -> List[Dict]:
        """
        执行版面检测

        Args:
            images: 单张或多张图像
            threshold: 置信度阈值

        Returns:
            检测结果列表
        """
        if threshold is not None:
            original_threshold = self.threshold
            self.threshold = threshold

        if isinstance(images, np.ndarray):
            images = [images]

        results = []
        for image in images:
            # 预处理
            input_dict, scale, ori_h, ori_w = self.preprocess(image)

            # 推理
            outputs = self.session.run(self.output_names, input_dict)

            # 后处理
            result = self.postprocess(image, outputs, scale, ori_h, ori_w)
            results.append(result)

        if threshold is not None:
            self.threshold = original_threshold

        return results


# ==================== UniRec ONNX Inference ====================
class UniRecONNXInference:
    """ONNX-based inference for UniRec model."""

    def __init__(
            self,
            encoder_path: str,
            decoder_path: str,
            mapping_path: str,
            device: str = 'cpu',  # 'auto', 'cuda', 'cpu'
    ):
        """Initialize ONNX inference sessions."""
        logger.info('Loading UniRec ONNX models...')

        # Determine execution providers based on device
        # 设置ONNX Runtime的执行提供者
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        # logger.info(f'   Using providers: {providers}')

        # Create ONNX runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.decoder_session = ort.InferenceSession(decoder_path,
                                                    sess_options,
                                                    providers=providers)
        self.encoder_session = ort.InferenceSession(encoder_path,
                                                    sess_options,
                                                    providers=providers)

        # Log actual provider being used
        _ = self.encoder_session.get_providers()[0]  # noqa: F841
        # logger.info(f'   Active provider: {actual_provider}')

        # Initialize processor and tokenizer
        self.processor = SimpleImageProcessor()
        self.tokenizer = SimpleTokenizer(mapping_file=mapping_path)

        # Get model info from decoder session
        self.num_decoder_layers = None
        self.num_heads = None
        self.head_dim = None

        for inp in self.decoder_session.get_inputs():
            if 'past_key' in inp.name:
                layer_idx = int(inp.name.split('_')[-1])
                if self.num_decoder_layers is None or layer_idx + 1 > self.num_decoder_layers:
                    self.num_decoder_layers = layer_idx + 1
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

        logger.info('✅ UniRec models loaded successfully!')
        # logger.info(f'   Number of decoder layers: {self.num_decoder_layers}')
        # logger.info(f'   Number of attention heads: {self.num_heads}')
        # logger.info(f'   Head dimension: {self.head_dim}')
        # logger.info(f'   Model dimension (d_model): {self.d_model}')
        # logger.info(f'   Vocabulary size: {self.tokenizer.vocab_size}')

    def encode_image(self, image):
        """Encode image using encoder ONNX model."""
        data_img = self.processor(image)
        pixel_values = data_img['pixel_values']

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
        input_ids = np.array([[input_id]], dtype=np.int64)
        position_ids = np.array([[padding_idx + 1 + past_length]],
                                dtype=np.int64)

        decoder_inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'cross_k': cross_k.astype(np.float32),
            'cross_v': cross_v.astype(np.float32),
        }

        for i, (past_key, past_value) in enumerate(past_key_values):
            decoder_inputs[f'past_key_{i}'] = past_key.astype(np.float32)
            decoder_inputs[f'past_value_{i}'] = past_value.astype(np.float32)

        decoder_outputs = self.decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]

        present_key_values = []
        for i in range(self.num_decoder_layers):
            key = decoder_outputs[1 + i * 2]
            value = decoder_outputs[1 + i * 2 + 1]
            present_key_values.append((key, value))

        return logits, present_key_values

    def generate(self,
                 image,
                 max_length=2048,
                 bos_token_id=None,
                 eos_token_id=None,
                 pad_token_id=None):
        """Generate text from image."""
        if bos_token_id is None:
            bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        encoder_hidden_states, cross_k, cross_v = self.encode_image(image)
        generated_ids = [bos_token_id]

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

        for step in range(max_length - 1):
            current_token = generated_ids[-1]
            past_length = step

            logits, past_key_values = self.decode_step(
                current_token,
                past_length,
                cross_k,
                cross_v,
                past_key_values,
                padding_idx=pad_token_id)

            next_token_id = int(np.argmax(logits[0, -1, :]))
            generated_ids.append(next_token_id)

            if next_token_id == eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids,
                                               skip_special_tokens=False)
        cleaned_text = clean_special_tokens(generated_text)

        return cleaned_text, generated_ids


# ==================== OpenDoc ONNX Pipeline ====================
class OpenDocONNX:
    """完整的文档OCR ONNX Pipeline"""

    def __init__(
        self,
        layout_model_path: str,
        unirec_encoder_path: str,
        unirec_decoder_path: str,
        tokenizer_mapping_path: str,
        device: str = 'cpu',
        layout_threshold: float = 0.5,
        use_layout_detection: bool = True,
        use_chart_recognition: bool = True,
    ):
        """
        初始化OpenDoc ONNX Pipeline

        Args:
            layout_model_path: 版面检测ONNX模型路径
            unirec_encoder_path: UniRec编码器ONNX模型路径
            unirec_decoder_path: UniRec解码器ONNX模型路径
            tokenizer_mapping_path: Tokenizer映射文件路径
            device: 'cpu' 或 'cuda'
            layout_threshold: 版面检测阈值
            use_layout_detection: 是否使用版面检测
            use_chart_recognition: 是否识别图表
        """
        self.device = device
        self.use_layout_detection = use_layout_detection
        self.use_chart_recognition = use_chart_recognition

        # Markdown忽略的标签
        self.markdown_ignore_labels = [
            'number', 'footnote', 'header', 'header_image', 'footer',
            'footer_image', 'aside_text'
        ]

        # 初始化版面检测模型
        if use_layout_detection:
            self.layout_detector = LayoutDetectorONNX(
                layout_model_path, device=device, threshold=layout_threshold)
        else:
            self.layout_detector = None

        # 初始化VLM模型
        self.vlm_recognizer = UniRecONNXInference(
            encoder_path=unirec_encoder_path,
            decoder_path=unirec_decoder_path,
            mapping_path=tokenizer_mapping_path,
            device=device)

        logger.info('✅ OpenDoc ONNX Pipeline initialized successfully')

    def predict(
        self,
        image_path: str,
        layout_threshold: Optional[float] = None,
        max_length: int = 2048,
        merge_layout_blocks: bool = True,
    ) -> Dict:
        """
        对单张图像进行预测，仿照 get_layout_parsing_results 的逻辑

        Args:
            image_path: 图像路径
            layout_threshold: 版面检测阈值
            max_length: VLM最大生成长度
            merge_layout_blocks: 是否合并布局块

        Returns:
            预测结果字典
        """
        start_time = time.time()

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        ori_h, ori_w = image.shape[:2]

        # 版面检测
        layout_results = None
        if self.use_layout_detection:
            layout_results = self.layout_detector(
                [image], threshold=layout_threshold)[0]
        else:
            # 整张图作为一个区域
            layout_results = {
                'boxes': [{
                    'cls_id': 0,
                    'label': 'text',
                    'score': 1.0,
                    'coordinate': [0, 0, ori_w, ori_h]
                }]
            }
            logger.info('  Layout detection disabled, processing whole image')

        # 确定 image_labels
        image_labels = (IMAGE_LABELS if self.use_chart_recognition else
                        IMAGE_LABELS + ['chart'])

        # 裁剪图像区域并合并布局块
        boxes = layout_results['boxes']
        blocks = []
        for box in boxes:
            coord = box['coordinate']
            x1, y1, x2, y2 = map(int, coord)
            cropped_img = image[y1:y2, x1:x2]
            if cropped_img.size == 0:
                cropped_img = None
            blocks.append({
                'img': cropped_img,
                'box': coord,
                'label': box['label'],
                'score': box.get('score', 1.0),
            })

        # 合并布局块
        if merge_layout_blocks:
            blocks = merge_blocks(blocks,
                                  non_merge_labels=image_labels + ['table'])

        # 收集需要VLM处理的blocks
        block_imgs = []
        text_prompts = []
        block_labels = []
        vlm_block_ids = []
        figure_token_maps = []
        drop_figures_set = set()
        imgs_in_doc = []  # 当前图像中的图片区域

        for j, block in enumerate(blocks):
            block_label = block['label']
            # 提取基础标签名（去除编号后缀）
            base_label = block_label.rsplit(
                '_', 1)[0] if '_' in block_label and block_label.rsplit(
                    '_', 1)[1].isdigit() else block_label
            if base_label in image_labels and block['img'] is not None:
                x_min, y_min, x_max, y_max = list(map(int, block['box']))
                img_path = f'imgs/img_in_{base_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg'
                imgs_in_doc.append({
                    'coordinate': block['box'],
                    'path': img_path
                })

        # 处理每个block
        for j, block in enumerate(blocks):
            block_img = block['img']
            block_label = block['label']
            # 提取基础标签名（去除编号后缀）
            base_label = block_label.rsplit(
                '_', 1)[0] if '_' in block_label and block_label.rsplit(
                    '_', 1)[1].isdigit() else block_label

            if base_label not in image_labels and block_img is not None:
                figure_token_map = {}
                text_prompt = 'OCR:'
                drop_figures = []

                if 'table' in block_label:
                    text_prompt = 'Table Recognition:'
                    block_img, figure_token_map, drop_figures = (
                        tokenize_figure_of_table(block_img, block['box'],
                                                 imgs_in_doc))
                elif block_label == 'chart' and self.use_chart_recognition:
                    text_prompt = 'Chart Recognition:'
                elif 'formula' in block_label and block_label != 'formula_number':
                    text_prompt = 'Formula Recognition:'
                    block_img = crop_margin(block_img)

                block_imgs.append(block_img)
                text_prompts.append(text_prompt)
                block_labels.append(block_label)
                figure_token_maps.append(figure_token_map)
                vlm_block_ids.append(j)
                drop_figures_set.update(drop_figures)

        # VLM识别
        vl_rec_results = []

        for block_img, block_label in zip(block_imgs, block_labels):
            # 转换为RGB PIL Image
            block_img_rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(block_img_rgb)

            try:
                text, token_ids = self.vlm_recognizer.generate(
                    pil_image, max_length=max_length)
            except Exception as e:
                logger.error(f"  Error processing block: {e}")
                text = ''

            # 使用 markdown_converter 进行后处理
            if 'table' in block_label:
                text = markdown_converter._handle_table(text)
            elif 'formula' in block_label and block_label != 'formula_number':
                text = markdown_converter._handle_formula(text)
            else:
                text = markdown_converter._handle_text(text)

            vl_rec_results.append(text)

        # 组装
        recognition_results = []
        curr_vlm_block_idx = 0

        for j, block in enumerate(blocks):
            block_img = block['img']
            block_bbox = block['box']
            block_label = block['label']
            block_content = ''

            if curr_vlm_block_idx < len(
                    vlm_block_ids) and vlm_block_ids[curr_vlm_block_idx] == j:
                result_str = vl_rec_results[curr_vlm_block_idx]
                figure_token_map = figure_token_maps[curr_vlm_block_idx]
                curr_vlm_block_idx += 1

                if result_str is None:
                    result_str = ''

                # 截断重复内容
                result_str = truncate_repetitive_content(result_str)

                # 处理公式符号替换
                has_paren = '\\(' in result_str and '\\)' in result_str
                has_bracket = '\\[' in result_str and '\\]' in result_str
                if has_paren or has_bracket:
                    result_str = result_str.replace('$', '')
                    result_str = (result_str.replace('\\(', ' $ ').replace(
                        '\\)', ' $ ').replace('\\[',
                                              ' $$ ').replace('\\]', ' $$ '))
                    if block_label == 'formula_number':
                        result_str = result_str.replace('$', '')

                # 对 table 结果进行 OTSL 转 HTML 和 untokenize
                if 'table' in block_label:
                    html_str = convert_otsl_to_html(result_str)
                    if html_str != '':
                        result_str = html_str
                    result_str = untokenize_figure_of_table(
                        result_str, figure_token_map)

                block_content = result_str

            # 处理图像类标签（去除编号后缀判断）
            base_label = block_label.rsplit(
                '_', 1)[0] if '_' in block_label and block_label.rsplit(
                    '_', 1)[1].isdigit() else block_label
            if base_label in image_labels and block_img is not None:
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = f'imgs/img_in_{base_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg'
                if img_path in drop_figures_set:
                    continue
                recognition_results.append({
                    'label': block_label,
                    'bbox': block_bbox,
                    'score': block.get('score', 1.0),
                    'text': '',
                    'text_unirec': '',
                    'is_image': True,
                    'img_path': img_path
                })
            else:
                recognition_results.append({
                    'label': block_label,
                    'bbox': block_bbox,
                    'score': block.get('score', 1.0),
                    'text': block_content,
                    'text_unirec': block_content,
                    'is_image': False
                })

        total_time = time.time() - start_time
        logger.info(f"  Total time: {total_time: .3f}s")

        return {
            'input_path': image_path,
            'width': ori_w,
            'height': ori_h,
            'layout_results': layout_results,
            'recognition_results': recognition_results,
            'blocks': blocks,
            'timing': {
                'total': total_time,
            }
        }

    def save_to_json(self, result: Dict, output_path: str):
        """保存结果为JSON"""
        if 'layout_results' in result:
            del result['layout_results']

        # 移除包含 ndarray 的 blocks 字段，无法序列化为 JSON
        if 'blocks' in result:
            del result['blocks']

        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        json_path = os.path.join(img_dir, f"{img_name}.json")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # logger.info(f"  Saved JSON to {json_path}")

    def save_to_markdown(self, result: Dict, output_path: str):
        """保存结果为Markdown，按阅读顺序包含图片"""
        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        md_path = os.path.join(img_dir, f"{img_name}.md")

        # 创建imgs子目录
        imgs_dir = os.path.join(img_dir, 'imgs')
        os.makedirs(imgs_dir, exist_ok=True)

        # 读取原始图像用于裁剪保存图片
        original_image = cv2.imread(result['input_path'])
        ori_width = result.get(
            'width',
            original_image.shape[1] if original_image is not None else 1)

        with open(md_path, 'w', encoding='utf-8') as f:

            for rec in result['recognition_results']:
                # 获取基础标签名（去除编号后缀，如 text_01 -> text）
                label = rec['label']
                base_label = label.rsplit(
                    '_', 1)[0] if '_' in label and label.rsplit(
                        '_', 1)[1].isdigit() else label

                # 跳过忽略的标签
                if base_label in self.markdown_ignore_labels:
                    continue

                # 处理图片类型
                if rec.get('is_image', False):
                    img_path = rec.get('img_path', '')
                    if img_path and original_image is not None:
                        # 从bbox裁剪并保存图片
                        bbox = rec.get('bbox', [])
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            cropped_img = original_image[y1:y2, x1:x2]
                            if cropped_img.size > 0:
                                save_img_path = os.path.join(img_dir, img_path)
                                os.makedirs(os.path.dirname(save_img_path),
                                            exist_ok=True)
                                cv2.imwrite(save_img_path, cropped_img)
                            # 计算图片宽度占原图的百分比
                            img_width = x2 - x1
                            width_percent = int((img_width / ori_width) * 100)
                            width_percent = max(5, min(width_percent,
                                                       100))  # 限制在5%-100%之间
                        else:
                            width_percent = 50  # 默认50%
                        f.write(
                            f'<img src="{img_path}" alt="Image" width="{width_percent}%" />\n\n'
                        )
                    continue

                text = rec['text'].strip()
                if not text:
                    continue

                # 根据标签类型格式化输出
                if 'title' in base_label or base_label == 'doc_title':
                    f.write(f"## {text}\n\n")
                elif 'table' in base_label:
                    f.write(f"{text}\n\n")
                elif 'formula' in base_label or base_label == 'equation':
                    f.write(f"$${text}$$\n\n")
                else:
                    f.write(f"{text}\n\n")

        # logger.info(f"  Saved Markdown to {md_path}")

    def save_visualization(self, result: Dict, output_path: str):
        """保存可视化结果"""
        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        vis_path = os.path.join(img_dir, f"{img_name}_vis.jpg")

        image = cv2.imread(result['input_path'])

        colors = {
            'text': (0, 255, 0),
            'title': (255, 0, 0),
            'table': (0, 0, 255),
            'figure': (255, 255, 0),
            'formula': (255, 0, 255),
            'equation': (255, 0, 255),
        }

        for box_info in result['layout_results']['boxes']:
            x1, y1, x2, y2 = map(int, box_info['coordinate'])
            label = box_info['label']
            score = box_info['score']

            color = colors.get(label, (255, 0, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label}: {score: .2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(vis_path, image)
        # logger.info(f"  Saved visualization to {vis_path}")


# ==================== Main Function ====================
def main():
    desc = 'OpenDoc ONNX Pipeline - Full Document OCR with Layout Detection'
    parser = argparse.ArgumentParser(description=desc)

    # Input/Output
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_path',
                        type=str,
                        default='./output_onnx',
                        help='Path to save output results')

    # Model paths
    parser.add_argument('--layout_model',
                        type=str,
                        required=True,
                        help='Path to layout detection ONNX model')
    parser.add_argument('--encoder_model',
                        type=str,
                        required=True,
                        help='Path to UniRec encoder ONNX model')
    parser.add_argument('--decoder_model',
                        type=str,
                        required=True,
                        help='Path to UniRec decoder ONNX model')
    parser.add_argument('--tokenizer_mapping',
                        type=str,
                        required=True,
                        help='Path to tokenizer mapping JSON file')

    # Settings
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference')
    parser.add_argument('--layout_threshold',
                        type=float,
                        default=0.4,
                        help='Layout detection threshold')
    parser.add_argument('--max_length',
                        type=int,
                        default=2048,
                        help='Maximum generation length for VLM')
    parser.add_argument('--use_layout_detection',
                        action='store_true',
                        default=True,
                        help='Use layout detection')
    parser.add_argument('--no_layout_detection',
                        dest='use_layout_detection',
                        action='store_false',
                        help='Disable layout detection (process whole image)')
    parser.add_argument('--use_chart_recognition',
                        action='store_true',
                        default=True,
                        help='Recognize charts')

    # Output formats
    parser.add_argument('--save_vis',
                        action='store_true',
                        default=True,
                        help='Save visualization images')
    parser.add_argument('--save_json',
                        action='store_true',
                        default=True,
                        help='Save JSON results')
    parser.add_argument('--save_markdown',
                        action='store_true',
                        default=True,
                        help='Save Markdown results')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 初始化Pipeline
    logger.info('=' * 80)
    logger.info('Initializing OpenDoc ONNX Pipeline')
    logger.info('=' * 80)

    opendoc_onnx = OpenDocONNX(
        layout_model_path=args.layout_model,
        unirec_encoder_path=args.encoder_model,
        unirec_decoder_path=args.decoder_model,
        tokenizer_mapping_path=args.tokenizer_mapping,
        device=args.device,
        layout_threshold=args.layout_threshold,
        use_layout_detection=args.use_layout_detection,
        use_chart_recognition=args.use_chart_recognition,
    )

    # 获取图像列表
    img_list = get_image_file_list(args.input_path)
    logger.info(f'\nFound {len(img_list)} images in {args.input_path}')
    logger.info(f'Using device: {args.device}')
    logger.info(f'Output will be saved to: {args.output_path}')
    logger.info('=' * 80)

    # 处理每张图像
    for idx, img_path in enumerate(img_list):
        logger.info(
            f"\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}"
        )

        try:
            # 预测
            result = opendoc_onnx.predict(
                img_path,
                layout_threshold=args.layout_threshold,
                max_length=args.max_length,
            )

            # 保存结果
            if args.save_vis:
                opendoc_onnx.save_visualization(result, args.output_path)

            if args.save_json:
                opendoc_onnx.save_to_json(result, args.output_path)

            if args.save_markdown:
                opendoc_onnx.save_to_markdown(result, args.output_path)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logger.info('\n' + '=' * 80)
    logger.info(
        f'✅ All processing completed! Results saved to {args.output_path}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()

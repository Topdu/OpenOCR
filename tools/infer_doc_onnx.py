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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list

from tools.utils.opendoc_onnx_utils.utils import (
    convert_otsl_to_html,
    crop_margin,
    filter_overlap_boxes,
    merge_blocks,
    tokenize_figure_of_table,
    truncate_repetitive_content,
    untokenize_figure_of_table,
)
from tools.to_markdown import MarkdownConverter
from tools.infer_unirec_onnx import (
    UniRecONNX
)

# 创建全局 markdown_converter 实例
markdown_converter = MarkdownConverter()

logger = get_logger(name='opendoc_onnx')

root_dir = Path(__file__).resolve().parent

IMAGE_LABELS = ['image', 'header_image', 'footer_image', 'seal']


def download_layout_model(model_dir=None):
    """Download layout detection ONNX model from ModelScope or HuggingFace.

    Args:
        model_dir: Directory to save model file. If None, use default cache directory.

    Returns:
        Path to the downloaded model file
    """
    # Use default cache directory if not specified
    if model_dir is None:
        cache_dir = Path.home() / '.cache' / 'openocr'
        model_dir = cache_dir / 'PP_DoclayoutV2_onnx'
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = 'PP-DoclayoutV2.onnx'
    model_path = model_dir / model_file

    # Check if model already exists
    if model_path.exists():
        logger.info(f'✅ Layout model found in {model_dir}')
        return str(model_path)

    logger.info(f'📥 Downloading layout model to {model_dir}...')

    download_success = False

    try:
        # Try ModelScope first (default)
        logger.info('🌐 Trying ModelScope (China mirror) first...')
        try:
            from modelscope import snapshot_download
            downloaded_path = snapshot_download(
                'topdktu/PP_DoclayoutV2_onnx',
                cache_dir=str(model_dir.parent)
            )
            logger.info(f'✅ Downloaded to {downloaded_path}')

            # Copy file to target directory
            import shutil
            src = Path(downloaded_path) / model_file
            if src.exists() and not model_path.exists():
                shutil.copy(str(src), str(model_path))
                logger.info(f'  ✓ {model_file}')

            # Verify file exists after download
            if model_path.exists():
                download_success = True
                logger.info('✅ Layout model downloaded successfully from ModelScope!')
            else:
                logger.info('⚠️  ModelScope download incomplete, trying HuggingFace...')

        except ImportError:
            logger.info('ModelScope not installed. Install with: pip install modelscope')
            logger.info('Trying HuggingFace...')
        except Exception as e:
            logger.info(f'ModelScope download failed: {e}')
            logger.info('Trying HuggingFace...')

        if not download_success:
            # Try HuggingFace
            logger.info('🌐 Using HuggingFace...')
            try:
                from huggingface_hub import hf_hub_download
                logger.info(f'  Downloading {model_file}...')
                downloaded_path = hf_hub_download(
                    repo_id='topdu/PP_DoclayoutV2_onnx',
                    filename=model_file,
                    cache_dir=str(model_dir.parent),
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f'  ✓ {model_file}')

                # Verify file exists after download
                if model_path.exists():
                    download_success = True
                    logger.info('✅ Layout model downloaded successfully from HuggingFace!')

            except ImportError:
                raise ImportError('HuggingFace Hub not installed. Install with: pip install huggingface_hub')

        if not download_success:
            raise RuntimeError(
                'Failed to download layout model. Please manually download from:\n'
                '  - https://huggingface.co/topdu/PP_DoclayoutV2_onnx\n'
                '  - https://modelscope.cn/models/topdktu/PP_DoclayoutV2_onnx'
            )

    except Exception as e:
        logger.error(f'❌ Failed to download layout model: {e}')
        raise

    return str(model_path)


def check_and_download_layout_model(model_path, auto_download=True):
    """Check if layout model exists, download if missing.

    Args:
        model_path: Path to layout model file
        auto_download: If True, automatically download missing model

    Returns:
        Path to the model file
    """
    if model_path and os.path.exists(model_path):
        return model_path

    if not auto_download:
        if not model_path or not os.path.exists(model_path):
            logger.error(f'⚠️  Layout model not found: {model_path}')
            logger.info('\n📝 Manual download instructions:')
            logger.info('   1. Visit: https://huggingface.co/topdu/PP_DoclayoutV2_onnx')
            logger.info('   2. Download PP-DoclayoutV2.onnx')
            logger.info('   3. Specify path with --layout_model argument')
            raise FileNotFoundError(f'Layout model not found: {model_path}')

    # Determine model directory from model path
    default_path = str(Path.home() / '.cache' / 'openocr' / 'PP_DoclayoutV2_onnx' / 'PP-DoclayoutV2.onnx')
    if model_path and model_path != default_path:
        # User specified a custom path
        model_dir = os.path.dirname(model_path)
    else:
        # Use default cache directory
        model_dir = None

    # Try ModelScope first (faster in China), then HuggingFace
    try:
        logger.info('🇨🇳 Trying ModelScope (China mirror) first...')
        return download_layout_model(model_dir)
    except:
        logger.info('🌍 Trying HuggingFace...')
        return download_layout_model(model_dir)


def _get_image_name_and_dir(result: Dict, output_path: str):
    """根据图片名创建子目录并返回(img_name, img_dir)"""
    img_name = os.path.basename(result['input_path'])
    if '.' in img_name:
        img_name = img_name.rsplit('.', 1)[0]

    # For PDF pages, append page number to avoid overwriting
    if 'pdf_page' in result:
        img_name = f'{img_name}_page{result["pdf_page"]}'

    img_dir = os.path.join(output_path, img_name)
    os.makedirs(img_dir, exist_ok=True)

    return img_name, img_dir


# ==================== Layout Detection ONNX ====================
class LayoutDetectorONNX:
    """ONNX版本的版面检测模型"""

    def __init__(self,
                 model_path: str,
                 use_gpu: Optional[bool] = None,
                 threshold: float = 0.5,
                 auto_download: bool = True):
        """
        初始化ONNX版面检测模型

        Args:
            model_path: ONNX模型路径
            use_gpu: Whether to use GPU. If None, auto-detect. If True, force GPU. If False, force CPU.
            threshold: 检测阈值
            auto_download: If True, automatically download missing model
        """
        self.threshold = threshold

        # Check and download model if needed
        model_path = check_and_download_layout_model(model_path, auto_download=auto_download)

        # Determine execution providers
        providers = self._get_execution_providers(use_gpu)
        logger.info(f'Layout detector using: {providers[0]}')

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

        logger.info(f'   Input names: {self.input_names}')
        logger.info(f'   Output names: {self.output_names}')

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
            logger.info('🔧 User specified: Using CPU for layout detection')
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
                logger.info(f'🔧 User specified: Using GPU for layout detection ({gpu_providers[0]})')
                return gpu_providers + ['CPUExecutionProvider']
            else:
                logger.warning('⚠️  GPU requested but not available, falling back to CPU')
                return ['CPUExecutionProvider']

        # Auto-detect (use_gpu is None)
        if gpu_providers:
            logger.info(f'✅ GPU detected for layout detection: Using {gpu_providers[0]}')
            return gpu_providers + ['CPUExecutionProvider']
        else:
            logger.info('ℹ️  No GPU detected for layout detection, using CPU')
            return ['CPUExecutionProvider']



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

        # Use exact target size to avoid float truncation mismatch (e.g., 799 instead of 800).
        new_h, new_w = target_h, target_w
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
            box['label'] = f'{base_label}_{idx:02d}'

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



# ==================== OpenDoc ONNX Pipeline ====================
class OpenDocONNX:
    """完整的文档OCR ONNX Pipeline"""

    def __init__(
        self,
        layout_model_path: Optional[str] = None,
        unirec_encoder_path: Optional[str] = None,
        unirec_decoder_path: Optional[str] = None,
        tokenizer_mapping_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        layout_threshold: float = 0.5,
        use_layout_detection: bool = True,
        use_chart_recognition: bool = True,
        auto_download: bool = True,
        max_parallel_blocks: int = 4,
    ):
        """
        初始化OpenDoc ONNX Pipeline

        Args:
            layout_model_path: 版面检测ONNX模型路径. If None, use default cache directory.
            unirec_encoder_path: UniRec编码器ONNX模型路径. If None, use default cache directory.
            unirec_decoder_path: UniRec解码器ONNX模型路径. If None, use default cache directory.
            tokenizer_mapping_path: Tokenizer映射文件路径. If None, use default cache directory.
            use_gpu: Whether to use GPU. If None, auto-detect. If True, force GPU. If False, force CPU.
            layout_threshold: 版面检测阈值
            use_layout_detection: 是否使用版面检测
            use_chart_recognition: 是否识别图表
            auto_download: If True, automatically download missing models
            max_parallel_blocks: Maximum number of blocks to process in parallel for VLM recognition (default: 4)
        """
        self.use_layout_detection = use_layout_detection
        self.use_chart_recognition = use_chart_recognition
        self.max_parallel_blocks = max(1, max_parallel_blocks)

        # Set default paths if not provided
        if layout_model_path is None:
            cache_dir = Path.home() / '.cache' / 'openocr'
            layout_model_path = str(cache_dir / 'PP_DoclayoutV2_onnx' / 'PP-DoclayoutV2.onnx')

        # Markdown忽略的标签
        self.markdown_ignore_labels = [
            'number', 'footnote', 'header', 'footer', 'aside_text', 'footer_image', 'header_image', 'chart'
        ]

        # 为所有25种标签类型定义不同的颜色 (BGR格式)
        self.colors = {
            'abstract': (255, 128, 0),        # 橙色
            'algorithm': (128, 0, 255),       # 紫色
            'aside_text': (128, 128, 128),    # 灰色
            'chart': (0, 255, 255),           # 青色
            'content': (0, 255, 0),           # 绿色
            'display_formula': (255, 0, 255), # 品红
            'doc_title': (255, 0, 0),         # 红色
            'figure_title': (255, 128, 128),  # 浅红
            'footer': (64, 64, 64),           # 深灰
            'footer_image': (128, 64, 0),     # 棕色
            'footnote': (192, 192, 192),      # 浅灰
            'formula_number': (255, 128, 255),# 浅品红
            'header': (96, 96, 96),           # 中灰
            'header_image': (0, 128, 128),    # 深青
            'image': (0, 255, 255),           # 青色
            'inline_formula': (200, 0, 200),  # 深品红
            'number': (128, 255, 0),          # 黄绿
            'paragraph_title': (255, 64, 0),  # 橙红
            'reference': (0, 128, 255),       # 天蓝
            'reference_content': (128, 192, 255), # 浅蓝
            'seal': (0, 0, 128),              # 深蓝
            'table': (0, 0, 255),             # 蓝色
            'text': (0, 200, 0),              # 深绿
            'vertical_text': (128, 255, 128), # 浅绿
            'vision_footnote': (160, 160, 160) # 中浅灰
        }

        # 初始化版面检测模型
        if use_layout_detection:
            self.layout_detector = LayoutDetectorONNX(
                layout_model_path, use_gpu=use_gpu, threshold=layout_threshold, auto_download=auto_download)
        else:
            self.layout_detector = None

        # 初始化VLM模型 (shared across all parallel workers;
        # ONNX Runtime sessions are thread-safe)
        self.vlm_recognizer = UniRecONNX(
            encoder_path=unirec_encoder_path,
            decoder_path=unirec_decoder_path,
            mapping_path=tokenizer_mapping_path,
            use_gpu=use_gpu,
            auto_download=auto_download)

    def _recognize_single_block(
        self,
        block_img: np.ndarray,
        block_label: str,
        block_index: int,
        max_length: int,
    ) -> Tuple[int, str]:
        """Recognize a single block using the shared VLM recognizer.

        The underlying ONNX Runtime session is thread-safe, so multiple
        threads can call this method concurrently on the same session.

        Args:
            block_img: Block image in BGR format
            block_label: Block label string
            block_index: Original index in the block list (for preserving order)
            max_length: Maximum generation length

        Returns:
            Tuple of (block_index, recognized_text)
        """
        # Convert BGR to RGB PIL Image
        block_img_rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(block_img_rgb)

        try:
            text, token_ids = self.vlm_recognizer(
                image=pil_image, max_length=max_length)
        except Exception as e:
            logger.error(f'  Error processing block {block_index} ({block_label}): {e}')
            text = ''

        # Post-process with markdown_converter
        if 'table' in block_label:
            text = markdown_converter._handle_table(text)
        elif 'formula' in block_label and block_label != 'formula_number':
            text = markdown_converter._handle_formula(text)
        else:
            text = markdown_converter._handle_text(text)

        return block_index, text

    def _parallel_vlm_recognize(
        self,
        block_imgs: List[np.ndarray],
        block_labels: List[str],
        max_length: int,
    ) -> List[str]:
        """Run VLM recognition on multiple blocks in parallel.

        Uses ThreadPoolExecutor with multiple UniRecONNX instances to process
        up to max_parallel_blocks blocks simultaneously.

        Args:
            block_imgs: List of block images in BGR format
            block_labels: List of block label strings
            max_length: Maximum generation length

        Returns:
            List of recognized text strings (in original order)
        """
        num_blocks = len(block_imgs)
        if num_blocks == 0:
            return []

        # Determine effective parallelism
        num_workers = min(self.max_parallel_blocks, num_blocks)

        # If only 1 worker, fall back to sequential processing (no overhead)
        if num_workers <= 1:
            logger.info(f'  VLM recognition: processing {num_blocks} block(s) sequentially')
            results = []
            for i, (block_img, block_label) in enumerate(zip(block_imgs, block_labels)):
                _, text = self._recognize_single_block(
                    block_img, block_label, i, max_length)
                results.append(text)
            return results

        logger.info(f'  VLM recognition: processing {num_blocks} block(s) with {num_workers} parallel worker(s)')

        # Initialize result list with placeholders
        vl_rec_results = [''] * num_blocks

        # Process blocks in batches of num_workers
        for batch_start in range(0, num_blocks, num_workers):
            batch_end = min(batch_start + num_workers, num_blocks)
            batch_size = batch_end - batch_start
            logger.info(f'  Processing batch [{batch_start + 1}-{batch_end}] / {num_blocks}')

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_start, batch_end):
                    future = executor.submit(
                        self._recognize_single_block,
                        block_imgs[i],
                        block_labels[i],
                        i,
                        max_length,
                    )
                    futures[future] = i

                for future in as_completed(futures):
                    try:
                        block_index, text = future.result()
                        vl_rec_results[block_index] = text
                    except Exception as e:
                        block_index = futures[future]
                        logger.error(f'  Parallel VLM recognition failed for block {block_index}: {e}')
                        vl_rec_results[block_index] = ''

        return vl_rec_results

    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF file to a list of BGR numpy images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of numpy arrays in BGR format
        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                'PyMuPDF is required for PDF support. '
                'Install with: pip install PyMuPDF'
            )

        images = []
        with fitz.open(pdf_path) as pdf:
            for pg in range(pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                # If width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
        return images

    def __call__(
        self,
        img_path: Optional[str] = None,
        img_numpy: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        layout_threshold: Optional[float] = None,
        max_length: int = 2048,
        merge_layout_blocks: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """
        Unified interface for OpenDoc inference.

        Args:
            img_path: Path to input image or PDF file (str or Path)
            img_numpy: Input image as numpy array (BGR format)
            image_path: Alias for img_path (for backward compatibility)
            layout_threshold: Layout detection threshold
            max_length: VLM maximum generation length
            merge_layout_blocks: Whether to merge layout blocks

        Returns:
            Prediction result dictionary for single image input.
            List of prediction result dictionaries for PDF input (one per page).
        """
        # Handle backward compatibility: image_path is alias for img_path
        if image_path is not None and img_path is None:
            img_path = image_path

        # Handle PDF input: convert to images and process each page
        if img_path is not None and str(img_path).lower().endswith('.pdf'):
            logger.info(f'Processing PDF file: {img_path}')
            pdf_images = self._pdf_to_images(img_path)
            logger.info(f'Found {len(pdf_images)} pages in PDF')
            results = []
            for page_idx, page_img in enumerate(pdf_images):
                logger.info(f'\n--- Processing page {page_idx + 1}/{len(pdf_images)} ---')
                page_result = self._infer_single_image(
                    img_numpy=page_img,
                    original_path=img_path,
                    page_index=page_idx,
                    layout_threshold=layout_threshold,
                    max_length=max_length,
                    merge_layout_blocks=merge_layout_blocks,
                )
                results.append(page_result)
            return results

        # Load image from path or numpy array
        is_temp_file = False
        if img_path is not None:
            actual_path = img_path
        elif img_numpy is not None:
            # For numpy array input, we need to save it temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, img_numpy)
            actual_path = tmp_path
            is_temp_file = True
        else:
            raise ValueError('Either img_path or img_numpy must be provided')

        start_time = time.time()

        # 读取图像
        image = cv2.imread(actual_path)
        if image is None:
            raise ValueError(f'Failed to read image: {actual_path}')

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

        # VLM识别 (parallel processing with up to max_parallel_blocks workers)
        vl_rec_results = self._parallel_vlm_recognize(
            block_imgs, block_labels, max_length)

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

            # 判断是否是合并块的后续部分（img 为 None 表示是合并块的后续部分）
            is_merged_continuation = block_img is None

            if base_label in image_labels and block_img is not None:
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = f'imgs/img_in_{base_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg'
                # 不跳过表格中的图片，需要保存它们
                # if img_path in drop_figures_set:
                #     continue
                recognition_results.append({
                    'label': block_label,
                    'bbox': block_bbox,
                    'score': block.get('score', 1.0),
                    'text': '',
                    'text_unirec': '',
                    'is_image': True,
                    'img_path': img_path,
                    'is_merged_continuation': False,
                    'in_table': img_path in drop_figures_set  # 标记是否在表格中
                })
            else:
                recognition_results.append({
                    'label': block_label,
                    'bbox': block_bbox,
                    'score': block.get('score', 1.0),
                    'text': block_content,
                    'text_unirec': block_content,
                    'is_image': False,
                    'is_merged_continuation': is_merged_continuation
                })

        total_time = time.time() - start_time
        logger.info(f'  Total time: {total_time: .3f}s')

        result = {
            'input_path': actual_path if not is_temp_file else '<numpy_array>',
            'width': ori_w,
            'height': ori_h,
            'layout_results': layout_results,
            'recognition_results': recognition_results,
            'blocks': blocks,
            'timing': {
                'total': total_time,
            }
        }

        # Clean up temporary file if created
        if is_temp_file and os.path.exists(actual_path):
            os.remove(actual_path)

        return result

    def _infer_single_image(
        self,
        img_numpy: np.ndarray,
        original_path: str,
        page_index: int,
        layout_threshold: Optional[float] = None,
        max_length: int = 2048,
        merge_layout_blocks: bool = True,
    ) -> Dict:
        """Run inference on a single BGR numpy image (used for PDF pages).

        Args:
            img_numpy: Input image as numpy array (BGR format)
            original_path: Original PDF file path (for result metadata)
            page_index: Page index in the PDF
            layout_threshold: Layout detection threshold
            max_length: VLM maximum generation length
            merge_layout_blocks: Whether to merge layout blocks

        Returns:
            Prediction result dictionary
        """
        import tempfile
        # Save numpy image to a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img_numpy)

        try:
            # Reuse the main __call__ method with the temp image path
            result = self(
                img_path=tmp_path,
                layout_threshold=layout_threshold,
                max_length=max_length,
                merge_layout_blocks=merge_layout_blocks,
            )
            # Update input_path to indicate PDF source
            pdf_name = os.path.basename(original_path)
            result['input_path'] = f'{original_path}'
            result['pdf_page'] = page_index + 1
            result['pdf_source'] = pdf_name
            # Store the page image for save_to_markdown and save_visualization
            result['_page_image'] = img_numpy.copy()
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def save_to_json(self, result: Dict, output_path: str):
        """保存结果为JSON"""
        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        json_path = os.path.join(img_dir, f'{img_name}.json')

        # Create a filtered copy excluding non-serializable and internal fields
        exclude_keys = {'layout_results', 'blocks', '_page_image'}
        result_filtered = {k: v for k, v in result.items() if k not in exclude_keys}

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_filtered, f, ensure_ascii=False, indent=2)

        # logger.info(f"  Saved JSON to {json_path}")

    def save_to_markdown(self, result: Dict, output_path: str):
        """保存结果为Markdown，按阅读顺序包含图片"""
        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        md_path = os.path.join(img_dir, f'{img_name}.md')

        # 创建imgs子目录
        imgs_dir = os.path.join(img_dir, 'imgs')
        os.makedirs(imgs_dir, exist_ok=True)

        # 读取原始图像用于裁剪保存图片
        if '_page_image' in result:
            original_image = result['_page_image']
        else:
            original_image = cv2.imread(result['input_path'])
        ori_width = result.get(
            'width',
            original_image.shape[1] if original_image is not None else 1)

        # 保存所有图片区域（包括表格中的图片）
        if original_image is not None:
            for rec in result['recognition_results']:
                if rec.get('is_image', False):
                    img_path = rec.get('img_path', '')
                    if img_path:
                        bbox = rec.get('bbox', [])
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            cropped_img = original_image[y1:y2, x1:x2]
                            if cropped_img.size > 0:
                                save_img_path = os.path.join(img_dir, img_path)
                                os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                                cv2.imwrite(save_img_path, cropped_img)

        with open(md_path, 'w', encoding='utf-8') as f:
            pending_text = []  # 用于收集合并块的文本
            pending_label = None  # 当前合并块的标签类型

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
                    # 先输出之前累积的文本
                    if pending_text:
                        self._write_merged_text(f, pending_text, pending_label)
                        pending_text = []
                        pending_label = None

                    # 如果图片在表格中，跳过在markdown中独立显示（已在表格HTML中引用）
                    if rec.get('in_table', False):
                        continue

                    img_path = rec.get('img_path', '')
                    if img_path:
                        # 计算图片宽度占原图的百分比
                        bbox = rec.get('bbox', [])
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            img_width = x2 - x1
                            width_percent = int((img_width / ori_width) * 100)
                            width_percent = max(5, min(width_percent, 100))  # 限制在5%-100%之间
                        else:
                            width_percent = 50  # 默认50%
                        f.write(
                            f'<img src="{img_path}" alt="Image" width="{width_percent}%" />\n'
                        )
                    continue
                text = rec['text'].strip()
                if not text:
                    continue

                # 检查是否是合并块的后续部分
                is_merged_continuation = rec.get('is_merged_continuation', False)

                if is_merged_continuation and pending_text:
                    # 是合并块的后续部分，追加文本
                    pending_text.append(text)
                else:
                    # 先输出之前累积的文本
                    if pending_text:
                        self._write_merged_text(f, pending_text, pending_label)
                        pending_text = []
                        pending_label = None

                    # 开始新的文本块
                    pending_text.append(text)
                    pending_label = base_label

            # 输出最后累积的文本
            if pending_text:
                self._write_merged_text(f, pending_text, pending_label)

    def _write_merged_text(self, f, texts: List[str], base_label: str):
        """将合并的文本写入文件"""
        merged_text = ' '.join(texts)

        # 根据标签类型格式化输出
        if 'title' in base_label and base_label != 'figure_title':
            f.write(f'## {merged_text}\n\n')
        elif 'table' in base_label:
            f.write(f'{merged_text}\n\n')
        elif 'formula' in base_label or base_label == 'equation':
            f.write(f'{merged_text}\n\n')
        else:
            f.write(f'{merged_text}\n\n')

    def save_visualization(self, result: Dict, output_path: str):
        """保存可视化结果"""
        img_name, img_dir = _get_image_name_and_dir(result, output_path)
        vis_path = os.path.join(img_dir, f'{img_name}_vis.jpg')

        if '_page_image' in result:
            image = result['_page_image'].copy()
        else:
            image = cv2.imread(result['input_path'])

        for box_info in result['layout_results']['boxes']:
            x1, y1, x2, y2 = map(int, box_info['coordinate'])
            label = box_info['label']
            score = box_info['score']

            # 提取基础标签名（去除编号后缀，如 text_01 -> text）
            base_label = label.rsplit('_', 1)[0] if '_' in label and label.rsplit('_', 1)[1].isdigit() else label

            # 获取颜色，如果没有定义则使用默认红色
            color = self.colors.get(base_label, (255, 0, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{label}: {score: .2f}', (x1, y1 - 10),
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
                        default=None,
                        help='Path to layout detection ONNX model (default: ~/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx)')
    parser.add_argument('--encoder_model',
                        type=str,
                        default=None,
                        help='Path to UniRec encoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx)')
    parser.add_argument('--decoder_model',
                        type=str,
                        default=None,
                        help='Path to UniRec decoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx)')
    parser.add_argument('--tokenizer_mapping',
                        type=str,
                        default=None,
                        help='Path to tokenizer mapping JSON file (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json)')

    # Settings
    parser.add_argument('--use-gpu',
                        type=str,
                        default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Use GPU for inference (auto: auto-detect, true: force GPU, false: force CPU)')
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
                        help='Use layout detection')
    parser.add_argument('--no_layout_detection',
                        dest='use_layout_detection',
                        action='store_false',
                        help='Disable layout detection (process whole image)')
    parser.add_argument('--use_chart_recognition',
                        action='store_true',
                        help='Recognize charts')
    parser.add_argument('--no-auto-download',
                        action='store_true',
                        help='Disable automatic model download')
    parser.add_argument('--max_parallel_blocks',
                        type=int,
                        default=4,
                        help='Max parallel blocks for VLM recognition (default: 4)')

    # Output formats
    parser.add_argument('--save_vis',
                        action='store_true',
                        help='Save visualization images')
    parser.add_argument('--save_json',
                        action='store_true',
                        help='Save JSON results')
    parser.add_argument('--save_markdown',
                        action='store_true',
                        help='Save Markdown results')

    args = parser.parse_args()

    # Parse use_gpu argument
    if args.use_gpu == 'auto':
        use_gpu = None
    elif args.use_gpu == 'true':
        use_gpu = True
    else:
        use_gpu = False

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    opendoc_onnx = OpenDocONNX(
        layout_model_path=args.layout_model,
        unirec_encoder_path=args.encoder_model,
        unirec_decoder_path=args.decoder_model,
        tokenizer_mapping_path=args.tokenizer_mapping,
        use_gpu=use_gpu,
        layout_threshold=args.layout_threshold,
        use_layout_detection=args.use_layout_detection,
        use_chart_recognition=args.use_chart_recognition,
        auto_download=not args.no_auto_download,
        max_parallel_blocks=args.max_parallel_blocks,
    )

    # 获取图像列表
    img_list = get_image_file_list(args.input_path)
    logger.info(f'\nFound {len(img_list)} images in {args.input_path}')
    logger.info(f'Output will be saved to: {args.output_path}')
    logger.info('=' * 80)

    # 处理每张图像
    for idx, img_path in enumerate(img_list):
        logger.info(
            f'\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}'
        )

        try:
            # 预测
            result = opendoc_onnx(
                img_path=img_path,
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
            logger.error(f'Error processing {img_path}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

    logger.info('\n' + '=' * 80)
    logger.info(
        f'✅ All processing completed! Results saved to {args.output_path}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()

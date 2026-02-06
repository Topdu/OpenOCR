"""
OpenOCR Unified Interface
Provides a single entry point for all OCR tasks with task-based dispatching.

Supported tasks:
- 'ocr': End-to-end OCR (detection + recognition)
- 'det': Text detection only
- 'rec': Text recognition only
- 'unirec': Universal recognition with VLM
- 'doc': Document OCR with layout analysis
- 'launch_openocr_demo': Launch OpenOCR Gradio demo
- 'launch_unirec_demo': Launch UniRec Gradio demo
- 'launch_opendoc_demo': Launch OpenDoc Gradio demo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from typing import Optional, Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.utils.logging import get_logger

logger = get_logger(name='openocr_unified')


class OpenOCR:
    """
    Unified OpenOCR interface that dispatches to different task implementations.

    Supported tasks:
    - 'det': Text detection only
    - 'rec': Text recognition only
    - 'ocr': End-to-end OCR (text detection + recognition)
    - 'unirec': Universal recognition with Vision-Language Model
    - 'doc': Document OCR with layout analysis (tables, formulas, etc.)
    - 'launch_openocr_demo': Launch OpenOCR Gradio demo
    - 'launch_unirec_demo': Launch UniRec Gradio demo
    - 'launch_opendoc_demo': Launch OpenDoc Gradio demo
    """

    def __init__(
        self,
        task: str = 'ocr',
        # Common parameters
        use_gpu: str = 'auto',
        # OCR task parameters
        mode: str = 'mobile',
        backend: str = 'onnx',
        onnx_det_model_path: Optional[str] = None,
        onnx_rec_model_path: Optional[str] = None,
        drop_score: float = 0.5,
        det_box_type: str = 'quad',
        # UniRec task parameters
        unirec_encoder_path: Optional[str] = None,
        unirec_decoder_path: Optional[str] = None,
        tokenizer_mapping_path: Optional[str] = None,
        max_length: int = 2048,
        # Doc task parameters
        layout_model_path: Optional[str] = None,
        layout_threshold: float = 0.5,
        use_layout_detection: bool = True,
        use_chart_recognition: bool = True,
        auto_download: bool = True,
    ):
        """
        Initialize OpenOCR unified interface.

        Args:
            task: Task type ('ocr', 'det', 'rec', 'unirec', 'doc', 'launch_openocr_demo', 'launch_unirec_demo', 'launch_opendoc_demo')

            # Common parameters
            use_gpu: GPU usage strategy ('auto', 'true', or 'false')

            # OCR task parameters
            mode: Model mode ('mobile' or 'server')
            backend: Backend type ('onnx')
            onnx_det_model_path: Path to detection ONNX model
            onnx_rec_model_path: Path to recognition ONNX model
            drop_score: Score threshold for filtering results
            det_box_type: Detection box type ('quad' or 'poly')

            # UniRec task parameters
            unirec_encoder_path: Path to UniRec encoder ONNX model
            unirec_decoder_path: Path to UniRec decoder ONNX model
            tokenizer_mapping_path: Path to tokenizer mapping JSON
            max_length: Maximum generation length

            # Doc task parameters
            layout_model_path: Path to layout detection model
            layout_threshold: Layout detection threshold
            use_layout_detection: Whether to use layout detection
            use_chart_recognition: Whether to recognize charts
            auto_download: Whether to auto-download missing models
        """
        self.task = task.lower()
        self.model = None

        # Validate task
        valid_tasks = ['det', 'rec', 'ocr', 'unirec', 'doc', 'launch_openocr_demo', 'launch_unirec_demo', 'launch_opendoc_demo']
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Must be one of {valid_tasks}")

        logger.info(f"Initializing OpenOCR with task: {self.task}")

        # Demo tasks don't need model initialization
        if self.task in ['launch_openocr_demo', 'launch_unirec_demo', 'launch_opendoc_demo']:
            logger.info(f"Demo task '{self.task}' will be launched via command line")
            return

        # Initialize task-specific model
        if self.task == 'det':
            self._init_det_task(
                backend=backend,
                onnx_model_path=onnx_det_model_path,
                use_gpu=use_gpu
            )
        elif self.task == 'rec':
            self._init_rec_task(
                mode=mode,
                backend=backend,
                onnx_model_path=onnx_rec_model_path,
                use_gpu=use_gpu
            )
        elif self.task == 'ocr':
            self._init_ocr_task(
                mode=mode,
                backend=backend,
                onnx_det_model_path=onnx_det_model_path,
                onnx_rec_model_path=onnx_rec_model_path,
                drop_score=drop_score,
                det_box_type=det_box_type,
                use_gpu=use_gpu
            )
        elif self.task == 'unirec':
            self._init_unirec_task(
                encoder_path=unirec_encoder_path,
                decoder_path=unirec_decoder_path,
                mapping_path=tokenizer_mapping_path,
                use_gpu=use_gpu,
                auto_download=auto_download
            )
        elif self.task == 'doc':
            self._init_doc_task(
                layout_model_path=layout_model_path,
                unirec_encoder_path=unirec_encoder_path,
                unirec_decoder_path=unirec_decoder_path,
                tokenizer_mapping_path=tokenizer_mapping_path,
                use_gpu=use_gpu,
                layout_threshold=layout_threshold,
                use_layout_detection=use_layout_detection,
                use_chart_recognition=use_chart_recognition,
                auto_download=auto_download
            )

        logger.info(f"✅ OpenOCR initialized successfully for task: {self.task}")

    def _init_det_task(self, **kwargs):
        """Initialize detection task"""
        from tools.infer_det import OpenDetector
        self.model = OpenDetector(**kwargs)

    def _init_rec_task(self, **kwargs):
        """Initialize recognition task"""
        from tools.infer_rec import OpenRecognizer
        self.model = OpenRecognizer(**kwargs)

    def _init_ocr_task(self, **kwargs):
        """Initialize OCR task (detection + recognition)"""
        from tools.infer_e2e import OpenOCRE2E
        self.model = OpenOCRE2E(**kwargs)

    def _init_unirec_task(self, **kwargs):
        """Initialize UniRec task (universal recognition)"""
        from tools.infer_unirec_onnx import UniRecONNX
        self.model = UniRecONNX(**kwargs)

    def _init_doc_task(self, **kwargs):
        """Initialize Doc task (document OCR with layout)"""
        from tools.infer_doc_onnx import OpenDocONNX
        self.model = OpenDocONNX(**kwargs)

    def __call__(self, *args, **kwargs):
        """
        Execute the task with appropriate parameters.

        For 'det' task:
            Args:
                image_path: Path to image or directory
                return_mask: Whether to return detection mask

        For 'rec' task:
            Args:
                image_path: Path to image or directory
                batch_num: Batch size for recognition

        For 'ocr' task:
            Args:
                image_path: Path to image or directory
                is_visualize: Whether to visualize results
                rec_batch_num: Batch size for recognition
                crop_infer: Whether to use crop inference
                return_mask: Whether to return detection mask

        For 'unirec' task:
            Args:
                image_path: Path to image
                max_length: Maximum generation length

        For 'doc' task:
            Args:
                image_path: Path to image
                layout_threshold: Layout detection threshold
                max_length: Maximum generation length
                merge_layout_blocks: Whether to merge layout blocks

        Returns:
            Task-specific results
        """
        if self.model is None:
            raise RuntimeError('Model not initialized')

        # Dispatch to appropriate task
        if self.task == 'det':
            return self._call_det(*args, **kwargs)
        elif self.task == 'rec':
            return self._call_rec(*args, **kwargs)
        elif self.task == 'ocr':
            return self._call_ocr(*args, **kwargs)
        elif self.task == 'unirec':
            return self._call_unirec(*args, **kwargs)
        elif self.task == 'doc':
            return self._call_doc(*args, **kwargs)

    def _call_det(self, image_path, **kwargs):
        """Call detection task"""
        return self.model(img_path=image_path, **kwargs)

    def _call_rec(self, image_path, batch_num=1, **kwargs):
        """Call recognition task"""
        return self.model(img_path=image_path, batch_num=batch_num, **kwargs)

    def _call_ocr(self, image_path, **kwargs):
        """Call OCR task"""
        return self.model(img_path=image_path, **kwargs)

    def _call_unirec(self, image_path, max_length=2048, **kwargs):
        """Call UniRec task"""
        return self.model(img_path=image_path, max_length=max_length, **kwargs)

    def _call_doc(self, image_path, **kwargs):
        """Call Doc task"""
        return self.model(img_path=image_path, **kwargs)

    # Additional methods for doc task
    def save_to_json(self, result: Dict, output_path: str):
        """Save doc task results to JSON (only for doc task)"""
        if self.task != 'doc':
            raise RuntimeError("save_to_json is only available for 'doc' task")
        return self.model.save_to_json(result, output_path)

    def save_to_markdown(self, result: Dict, output_path: str):
        """Save doc task results to Markdown (only for doc task)"""
        if self.task != 'doc':
            raise RuntimeError("save_to_markdown is only available for 'doc' task")
        return self.model.save_to_markdown(result, output_path)

    def save_visualization(self, result: Dict, output_path: str):
        """Save doc task visualization (only for doc task)"""
        if self.task != 'doc':
            raise RuntimeError("save_visualization is only available for 'doc' task")
        return self.model.save_visualization(result, output_path)


def main():
    """Command-line interface for OpenOCR unified interface"""
    parser = argparse.ArgumentParser(
        description='OpenOCR Unified Interface - Single entry point for all OCR tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Command-line Usage:
  After installation, you can use OpenOCR in three ways:

  1. Using the 'openocr' command (recommended):
     openocr --task ocr --input_path image.jpg

  2. Using 'python -m openocr':
     python -m openocr --task ocr --input_path image.jpg

  3. Running the script directly:
     python openocr.py --task ocr --input_path image.jpg

Examples:
  # Detection task
  openocr --task det --input_path image.jpg

  # Recognition task
  openocr --task rec --input_path image.jpg --mode server

  # OCR task (detection + recognition)
  openocr --task ocr --input_path image.jpg --is_vis

  # OCR with custom output path
  openocr --task ocr --input_path ./images --output_path ./results

  # UniRec task (universal recognition)
  openocr --task unirec --input_path image.jpg --max_length 2048

  # Doc task (document OCR with layout)
  openocr --task doc --input_path document.jpg --save_markdown --save_json

  # Doc task with PDF input
  openocr --task doc --input_path document.pdf --save_markdown --save_json

  # Doc task with custom models
  openocr --task doc --input_path doc.jpg --layout_model path/to/layout.onnx \\
         --encoder_model path/to/encoder.onnx --decoder_model path/to/decoder.onnx

  # Launch OpenOCR Gradio demo
  openocr --task launch_openocr_demo --share

  # Launch UniRec Gradio demo
  openocr --task launch_unirec_demo --server_port 7861

  # Launch OpenDoc Gradio demo
  openocr --task launch_opendoc_demo --share --server_port 7862

For more information, visit: https://github.com/Topdu/OpenOCR
        """
    )

    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        default='ocr',
        choices=['det', 'rec', 'ocr', 'unirec', 'doc', 'launch_openocr_demo', 'launch_unirec_demo', 'launch_opendoc_demo'],
        help='Task type: det (detection), rec (recognition), ocr (detection+recognition), unirec (universal recognition), doc (document OCR), launch_*_demo (launch Gradio demo)'
    )

    # Unified input/output parameters
    parser.add_argument('--input_path', type=str, help='Input image/PDF path or directory (unified for all tasks, not required for demo tasks)')
    parser.add_argument('--output_path', type=str, help='Output directory (auto-generated as openocr_output/{task} if not specified)')

    # Demo launch parameters
    parser.add_argument('--share', action='store_true', help='[Demo] Create a public share link')
    parser.add_argument('--server_port', type=int, default=7860, help='[Demo] Server port (default: 7860)')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='[Demo] Server name (default: 0.0.0.0)')

    # Common parameters
    parser.add_argument(
        '--use_gpu',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='GPU usage strategy: auto (detect automatically), true (force GPU), false (force CPU)'
    )

    # OCR/Det/Rec task parameters
    parser.add_argument('--mode', type=str, default='mobile', choices=['mobile', 'server'], help='[OCR/Rec] Model mode')
    parser.add_argument('--backend', type=str, default='onnx', choices=['torch', 'onnx'], help='[OCR] Backend type')
    parser.add_argument('--onnx_det_model_path', type=str, help='[OCR] Detection ONNX model path')
    parser.add_argument('--onnx_rec_model_path', type=str, help='[OCR] Recognition ONNX model path')
    parser.add_argument('--drop_score', type=float, default=0.5, help='[OCR] Score threshold')
    parser.add_argument('--det_box_type', type=str, default='quad', choices=['quad', 'poly'], help='[Det/OCR] Box type')
    parser.add_argument('--is_vis', action='store_true', help='[Det/OCR] Visualize results')
    parser.add_argument('--rec_batch_num', type=int, default=6, help='[Rec/OCR] Recognition batch size')
    parser.add_argument('--return_mask', action='store_true', help='[Det] Return detection mask')

    # UniRec task parameters
    parser.add_argument('--encoder_model', type=str, help='[Doc/UniRec] Encoder ONNX model path')
    parser.add_argument('--decoder_model', type=str, help='[Doc/UniRec] Decoder ONNX model path')
    parser.add_argument('--mapping', type=str, help='[UniRec] Tokenizer mapping JSON path')
    parser.add_argument('--max_length', type=int, default=2048, help='[UniRec/Doc] Max generation length')

    # Doc task parameters
    parser.add_argument('--layout_model', type=str, help='[Doc] Layout detection model path')
    parser.add_argument('--tokenizer_mapping', type=str, help='[Doc] Tokenizer mapping path')
    parser.add_argument('--layout_threshold', type=float, default=0.4, help='[Doc] Layout detection threshold')
    parser.add_argument('--use_layout_detection', action='store_true', help='[Doc] Use layout detection')
    parser.add_argument('--no_layout_detection', dest='use_layout_detection', action='store_false', help='[Doc] Disable layout detection')
    parser.add_argument('--use_chart_recognition', action='store_true', help='[Doc] Recognize charts')
    parser.add_argument('--save_vis', action='store_true', help='[Doc] Save visualization')
    parser.add_argument('--save_json', action='store_true', help='[Doc] Save JSON results')
    parser.add_argument('--save_markdown', action='store_true', help='[Doc] Save Markdown results')
    parser.add_argument('--no_auto_download', action='store_true', help='Disable automatic model download')

    args = parser.parse_args()

    # use_gpu is already a string from argparse choices

    # Handle demo tasks
    if args.task == 'launch_openocr_demo':
        logger.info('Launching OpenOCR Gradio demo...')
        from demo_gradio import launch_demo
        launch_demo(
            share=args.share,
            server_port=args.server_port,
            server_name=args.server_name
        )
        return

    elif args.task == 'launch_unirec_demo':
        logger.info('Launching UniRec Gradio demo...')
        from demo_unirec import launch_demo
        launch_demo(
            encoder_path=args.encoder_model,
            decoder_path=args.decoder_model,
            mapping_path=args.mapping,
            use_gpu=args.use_gpu,
            auto_download=not args.no_auto_download,
            share=args.share,
            server_port=args.server_port,
            server_name=args.server_name
        )
        return

    elif args.task == 'launch_opendoc_demo':
        logger.info('Launching OpenDoc Gradio demo...')
        from demo_opendoc import launch_demo
        launch_demo(
            layout_model_path=args.layout_model,
            unirec_encoder_path=args.encoder_model,
            unirec_decoder_path=args.decoder_model,
            tokenizer_mapping_path=args.tokenizer_mapping,
            use_gpu=args.use_gpu,
            auto_download=not args.no_auto_download,
            share=args.share,
            server_port=args.server_port,
            server_name=args.server_name
        )
        return

    # Set default output directory if not specified
    if not args.output_path:
        args.output_path = f'openocr_output/{args.task}'

    # Use input_path as unified input
    if not args.input_path:
        parser.error('--input_path is required for all tasks')

    # Initialize unified interface
    try:
        if args.task == 'det':
            openocr = OpenOCR(
                task='det',
                backend=args.backend,
                onnx_det_model_path=args.onnx_det_model_path,
                use_gpu=args.use_gpu
            )

            from tools.utils.utility import get_image_file_list
            img_list = get_image_file_list(args.input_path)

            logger.info(f'\nFound {len(img_list)} images in {args.input_path}')
            logger.info(f'Output will be saved to: {args.output_path}')
            logger.info('=' * 80)

            os.makedirs(args.output_path, exist_ok=True)

            import json
            with open(os.path.join(args.output_path, 'det_results.txt'), 'w') as fout:
                for idx, img_path in enumerate(img_list):
                    logger.info(f"\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}")

                    try:
                        results = openocr(
                            image_path=img_path,
                            return_mask=args.return_mask
                        )

                        boxes = results[0]['boxes']
                        elapse = results[0]['elapse']

                        logger.info(f"   Found {len(boxes)} text regions, time: {elapse:.3f}s")

                        # Save results
                        dt_boxes_json = [{'points': box.tolist()} for box in boxes]
                        fout.write(f"{img_path}\t{json.dumps(dt_boxes_json)}\n")

                        # Visualize if requested
                        if args.is_vis:
                            import cv2
                            import numpy as np
                            src_img = cv2.imread(img_path)
                            for box in boxes:
                                box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
                                cv2.polylines(src_img, [box], True, color=(255, 255, 0), thickness=2)
                            vis_path = os.path.join(args.output_path, os.path.basename(img_path))
                            cv2.imwrite(vis_path, src_img)

                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

            logger.info('\n' + '=' * 80)
            logger.info(f'✅ Detection task completed. Results saved to {args.output_path}')
            logger.info('=' * 80)

        elif args.task == 'rec':
            openocr = OpenOCR(
                task='rec',
                mode=args.mode,
                backend=args.backend,
                onnx_rec_model_path=args.onnx_rec_model_path,
                use_gpu=args.use_gpu
            )

            from tools.utils.utility import get_image_file_list
            img_list = get_image_file_list(args.input_path)

            logger.info(f'\nFound {len(img_list)} images in {args.input_path}')
            logger.info(f'Output will be saved to: {args.output_path}')
            logger.info('=' * 80)

            os.makedirs(args.output_path, exist_ok=True)

            with open(os.path.join(args.output_path, 'rec_results.txt'), 'w') as fout:
                for idx, img_path in enumerate(img_list):
                    logger.info(f"\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}")

                    try:
                        results = openocr(
                            image_path=img_path,
                            batch_num=args.rec_batch_num
                        )

                        text = results[0]['text']
                        score = results[0]['score']
                        elapse = results[0]['elapse']

                        logger.info(f"   Text: {text}, Score: {score:.3f}, Time: {elapse:.3f}s")

                        fout.write(f"{img_path}\t{text}\t{score:.3f}\n")

                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

            logger.info('\n' + '=' * 80)
            logger.info(f'✅ Recognition task completed. Results saved to {args.output_path}')
            logger.info('=' * 80)

        elif args.task == 'ocr':
            openocr = OpenOCR(
                task='ocr',
                mode=args.mode,
                backend=args.backend,
                onnx_det_model_path=args.onnx_det_model_path,
                onnx_rec_model_path=args.onnx_rec_model_path,
                drop_score=args.drop_score,
                det_box_type=args.det_box_type,
                use_gpu=args.use_gpu
            )

            results, time_dicts = openocr(
                image_path=args.input_path,
                save_dir=args.output_path,
                is_visualize=args.is_vis,
                rec_batch_num=args.rec_batch_num
            )

            logger.info(f"✅ OCR task completed. Results saved to {args.output_path}")

        elif args.task == 'unirec':
            openocr = OpenOCR(
                task='unirec',
                unirec_encoder_path=args.encoder_model,
                unirec_decoder_path=args.decoder_model,
                tokenizer_mapping_path=args.mapping,
                use_gpu=args.use_gpu,
                max_length=args.max_length,
                auto_download=not args.no_auto_download
            )

            from tools.utils.utility import get_image_file_list
            img_list = get_image_file_list(args.input_path)

            logger.info(f'\nFound {len(img_list)} images in {args.input_path}')
            logger.info(f'Output will be saved to: {args.output_path}')
            logger.info('=' * 80)

            os.makedirs(args.output_path, exist_ok=True)

            import json
            with open(os.path.join(args.output_path, 'unirec_results.txt'), 'w') as fout:
                for idx, img_path in enumerate(img_list):
                    logger.info(f"\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}")

                    try:
                        result_text, generated_ids = openocr(
                            image_path=img_path,
                            max_length=args.max_length
                        )

                        logger.info(f"   Generated {len(generated_ids)} tokens")
                        logger.info(f"   Text: {result_text[:100]}..." if len(result_text) > 100 else f"   Text: {result_text}")

                        image_name = os.path.basename(img_path)
                        result_dict = {'text': result_text}
                        fout.write(f"{image_name}\t{json.dumps(result_dict, ensure_ascii=False)}\n")

                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

            logger.info('\n' + '=' * 80)
            logger.info(f'✅ UniRec task completed. Results saved to {args.output_path}')
            logger.info('=' * 80)

        elif args.task == 'doc':
            openocr = OpenOCR(
                task='doc',
                layout_model_path=args.layout_model,
                unirec_encoder_path=args.encoder_model,
                unirec_decoder_path=args.decoder_model,
                tokenizer_mapping_path=args.tokenizer_mapping,
                use_gpu=args.use_gpu,
                layout_threshold=args.layout_threshold,
                use_layout_detection=args.use_layout_detection,
                use_chart_recognition=args.use_chart_recognition,
                auto_download=not args.no_auto_download
            )

            from tools.utils.utility import get_image_file_list
            img_list = get_image_file_list(args.input_path)

            logger.info(f'\nFound {len(img_list)} images/PDFs in {args.input_path}')
            logger.info(f'Output will be saved to: {args.output_path}')
            logger.info('=' * 80)

            os.makedirs(args.output_path, exist_ok=True)

            for idx, img_path in enumerate(img_list):
                logger.info(f"\n[{idx + 1}/{len(img_list)}] Processing: {os.path.basename(img_path)}")

                try:
                    result = openocr(
                        image_path=img_path,
                        layout_threshold=args.layout_threshold,
                        max_length=args.max_length
                    )

                    if args.save_vis:
                        openocr.save_visualization(result, args.output_path)

                    if args.save_json:
                        openocr.save_to_json(result, args.output_path)

                    if args.save_markdown:
                        openocr.save_to_markdown(result, args.output_path)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            logger.info('\n' + '=' * 80)
            logger.info(f'✅ Doc task completed. Results saved to {args.output_path}')
            logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

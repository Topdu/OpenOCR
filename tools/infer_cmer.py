# main_inference.py
import os
import sys
import time
import importlib
import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))


from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.utility import get_image_file_list
from tools.utils.logging import get_logger

logger = get_logger()

# --------------------------------------------------------------------------- #
# Dynamic Import Helper
# --------------------------------------------------------------------------- #
def dynamic_import_class(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logger.error(f"Could not import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Could not find class {class_name} in module {module_path}: {e}")
        raise

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def build_tokenizer(config):
    """
    Dynamically build Tokenizer
    """
    if config is None:
        raise ValueError("Config is missing 'Tokenizer' section.")
    
    module_path = config.get('module')
    class_name = config.get('class')
    args = config.get('args', {})

    if not module_path or not class_name:
        raise ValueError("Tokenizer config must specify 'module' and 'class'")

    logger.info(f"Building tokenizer from {module_path}.{class_name}")
    BuilderClass = dynamic_import_class(module_path, class_name)

    builder = BuilderClass(**args)
    
    if hasattr(builder, 'build'):
        return builder.build()
    else:
        return builder

def build_model(config):
    if 'Backbone' in config:
        backbone_cfg = config['Backbone']
        full_name = backbone_cfg.get('name') 
        model_args = {k: v for k, v in backbone_cfg.items() if k != 'name'}
        
        if full_name:
            module_path = ".".join(full_name.split('.')[:-1])
            class_name = full_name.split('.')[-1]
        else:
            module_path = config.get('module')
            class_name = config.get('class')
    else:
        module_path = config.get('module')
        class_name = config.get('class')
        model_args = config.get('args', {})
    
    if not module_path or not class_name:
        raise ValueError("Config must specify model class path")

    ModelClass = dynamic_import_class(module_path, class_name)
    logger.info(f"Building model {class_name} from {module_path}...")

    if "CMER" in class_name:
        try:
            ConfigClass = dynamic_import_class(module_path, "CMERConfig")
        except:
            logger.warning(f"Could not find CMERConfig in {module_path}, trying generic initialization.")
            return ModelClass(**model_args)

        vision_args = model_args.get('vision_config', {})
        decoder_args = model_args.get('decoder_config', {})
        
        for k in ['vocab_size', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            if k in model_args:
                decoder_args[k] = model_args[k]

        model_config = ConfigClass(
            vision_config=vision_args,
            decoder_config=decoder_args
        )
        return ModelClass(config=model_config)
    else:
        return ModelClass(**model_args)

def build_processor(config, tokenizer):
    if config is None:
        raise ValueError("Config is missing 'PreProcess' section.")

    img_proc_cfg = config.get('ImageProcessor', {})
    img_proc_module = img_proc_cfg.get('module')
    img_proc_class = img_proc_cfg.get('class')
    img_proc_args = img_proc_cfg.get('args', {})
    
    if not img_proc_module or not img_proc_class:
         raise ValueError("PreProcess.ImageProcessor must specify 'module' and 'class'")

    ImageProcessorClass = dynamic_import_class(img_proc_module, img_proc_class)
    image_processor = ImageProcessorClass(**img_proc_args)
    
    proc_cfg = config.get('Processor', {})
    proc_module = proc_cfg.get('module')
    proc_class = proc_cfg.get('class')
    
    if not proc_module or not proc_class:
         raise ValueError("PreProcess.Processor must specify 'module' and 'class'")

    ProcessorClass = dynamic_import_class(proc_module, proc_class)
    processor = ProcessorClass(image_processor=image_processor, tokenizer=tokenizer)
    
    return processor

def build_postprocess(config, tokenizer):
    if config is None:
        raise ValueError("Config is missing 'PostProcess' section.")
    
    module_path = config.get('module')
    class_name = config.get('class')
    args = config.get('args', {})

    if not module_path or not class_name:
        raise ValueError("PostProcess must specify 'module' and 'class'")

    PostProcessClass = dynamic_import_class(module_path, class_name)
    return PostProcessClass(tokenizer=tokenizer, **args)

def build_metric(config):
    if config is None:
        logger.warning("Config is missing 'Metric' section. Metrics will not be calculated.")
        return None
    
    module_path = config.get('module')
    class_name = config.get('class')
    args = config.get('args', {})

    if not module_path or not class_name:
        raise ValueError("Metric must specify 'module' and 'class'")

    MetricClass = dynamic_import_class(module_path, class_name)
    return MetricClass(**args)

# --------------------------------------------------------------------------- #
# CMERPredictor Class
# --------------------------------------------------------------------------- #
class CMERPredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 1. Set device
        requested_device = cfg['Global']['device']
        if requested_device == 'gpu':
            requested_device = 'cuda'
        if requested_device == 'cuda' and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        logger.info(f"Inference device: {self.device}")

        # 2. Build Tokenizer 
        self.tokenizer = build_tokenizer(cfg.get('Global', {}).get('Tokenizer'))

        # 3. Build Processor
        self.processor = build_processor(cfg.get('PreProcess'), self.tokenizer)

        # 4. Build PostProcess
        self.post_process = build_postprocess(cfg.get('PostProcess'), self.tokenizer)

        # 5. Build Model
        arch_config = cfg['Architecture']
        # Inject tokenizer attributes into model configuration
        vocab_args = {
            'vocab_size': len(self.tokenizer),
            'pad_token_id': self.tokenizer.pad_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        if 'Backbone' in arch_config:
            if 'decoder_config' in arch_config['Backbone']:
                arch_config['Backbone']['decoder_config'].update(vocab_args)
            else:
                arch_config['Backbone'].update(vocab_args)
        else:
            if 'args' not in arch_config:
                arch_config['args'] = {}
            if 'decoder_config' in arch_config['args']:
                arch_config['args']['decoder_config'].update(vocab_args)
            else:
                arch_config['args'].update(vocab_args)

        self.model = build_model(arch_config)

        # 6. Load weights
        ckpt_path = cfg['Global']['infer_weight']
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found at {ckpt_path}")
        
        logger.info(f"Loading weights from {ckpt_path}")
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict_full = load_file(ckpt_path)
        else:
            state_dict_full = torch.load(ckpt_path, map_location='cpu')

        state_dict = state_dict_full['state_dict'] if 'state_dict' in state_dict_full else state_dict_full
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = cfg['Global'].get('max_text_length', 256)
        self.beam_size = cfg['Global'].get('beam_size', 1)

    def __call__(self, img_path):
        if not os.path.exists(img_path):
            return None

        t_start = time.time()
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.beam_size,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    return_only_new_tokens=True
                )

            pred_text_list = self.post_process(generated_ids)
            pred_text = pred_text_list[0] if pred_text_list else ""

            t_end = time.time()
            
            return {
                'text': pred_text,
                'elapse': t_end - t_start,
                'file': img_path
            }

        except Exception as e:
            logger.error(f"Inference failed for {img_path}: {e}")
            return None

# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def main(cfg):
    predictor = CMERPredictor(cfg)

    save_res_path = cfg['Global'].get('save_res_path', './output/infer_results')
    if Path(save_res_path).suffix:
        os.makedirs(os.path.dirname(save_res_path), exist_ok=True)
        res_file = save_res_path
    else:
        os.makedirs(save_res_path, exist_ok=True)
        res_file = os.path.join(save_res_path, 'rec_results.txt')
    
    infer_label_file = cfg['Global'].get('infer_label_file', None)
    infer_img_root = cfg['Global'].get('infer_img_root', None)
    
    inference_data = []

    if infer_label_file and os.path.exists(infer_label_file):
        logger.info(f"Loading inference data from label file: {infer_label_file}")

        with open(infer_label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 2:
                    parts = line.split(' ', 1)
                
                if len(parts) >= 2:
                    img_name = parts[0]
                    gt_text = parts[1]
                else:
                    img_name = parts[0]
                    gt_text = ""

                full_img_path = os.path.join(infer_img_root, img_name)
                inference_data.append((full_img_path, gt_text))
    else:
        if infer_img_root:
            logger.info(f"Scanning directory/file for images: {infer_img_root}")
            img_list = get_image_file_list(infer_img_root)
            for img_file in img_list:
                gt_text = "None"
                inference_data.append((img_file, gt_text))
        else:
            logger.error("No inference data found.")
            return
        
    logger.info(f"Found {len(inference_data)} samples to infer.")

    metric_calculator = build_metric(cfg.get('Metric'))
    if metric_calculator is None:
        logger.warning("No Metric config found, metrics will be skipped.")

    t_sum = 0
    sample_num = 0
    valid_count = 0
    
    metric_accumulator = defaultdict(float)

    with open(res_file, 'w', encoding='utf-8') as fout:
        fout.write("File\tPrediction\tGroundTruth\tBLEU\tROUGE-1\tROUGE-2\tROUGE-L\tEditDist\n")
        
        for img_file, gt_text in inference_data:
            res = predictor(img_file)
            if res:
                current_metrics = None
                
                # Calculate metrics if GT is available
                if gt_text != "None" and gt_text != "" and metric_calculator is not None:
                    current_metrics = metric_calculator.compute_single(
                        preds=[res['text']], 
                        labels=[gt_text]
                    )

                    for k, v in current_metrics.items():
                        metric_accumulator[k] += v
                        
                    valid_count += 1

                logger.info(f"[{sample_num}] {os.path.basename(img_file)}")
                logger.info(f"  GT      : {gt_text}")
                logger.info(f"  Pred    : {res['text']}")
                
                if current_metrics:
                    logger.info(f"  Metrics : "
                                f"BLEU={current_metrics['bleu']:.4f}, "
                                f"R-1={current_metrics['rouge1']:.4f}, "
                                f"R-2={current_metrics['rouge2']:.4f}, "
                                f"R-L={current_metrics['rougeL']:.4f}, "
                                f"ED={current_metrics['edit_distance']:.2f}")
                
                clean_pred = res['text'].replace('\n', '\\n').replace('\t', ' ')
                clean_gt = gt_text.replace('\n', '\\n').replace('\t', ' ')
                
                bleu_val = f"{current_metrics['bleu']:.4f}" if current_metrics else "N/A"
                r1_val = f"{current_metrics['rouge1']:.4f}" if current_metrics else "N/A"
                r2_val = f"{current_metrics['rouge2']:.4f}" if current_metrics else "N/A"
                rl_val = f"{current_metrics['rougeL']:.4f}" if current_metrics else "N/A"
                ed_val = f"{current_metrics['edit_distance']:.2f}" if current_metrics else "N/A"

                fout.write(f"{img_file}\t{clean_pred}\t{clean_gt}\t{bleu_val}\t{r1_val}\t{r2_val}\t{rl_val}\t{ed_val}\n")
                
                t_sum += res['elapse']
                sample_num += 1

    avg_time = t_sum / sample_num if sample_num > 0 else 0
    
    logger.info("-" * 50)
    logger.info(f"Inference Done.")
    logger.info(f"Total Samples: {sample_num}")
    logger.info(f"Valid GTs    : {valid_count}")
    logger.info(f"Avg Time     : {avg_time:.4f}s")
    
    if valid_count > 0 and metric_calculator is not None:
        logger.info("Average Metrics (Calculated as arithmetic mean of single scores):")
        display_order = ["bleu", "rouge1", "rouge2", "rougeL", "edit_distance"]
        for k in display_order:
            if k in metric_accumulator:
                avg_val = metric_accumulator[k] / valid_count
                logger.info(f"  Avg {k:<13}: {avg_val:.4f}")
            
    logger.info(f"Results saved to {res_file}")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
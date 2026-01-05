import os
import json
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import importlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dynamic_import_class(module_path, class_name):
    """Dynamically import a class."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {class_name} in module {module_path}: {e}")

class CMERDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None, epoch=0, task='formula_rec'):
        super(CMERDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        
        dataset_config = config[mode]['dataset']
        global_config = config['Global']

        # --- 1. Path Configuration ---
        self.data_dir = dataset_config['data_dir']
        self.label_file_list = dataset_config.get('label_file_list', [])
        
        if isinstance(self.label_file_list, str):
            self.label_file_list = [self.label_file_list]

        # Max text length
        self.max_length = dataset_config.get('max_text_length', 1024)

        # --- 2. Build Processor ---
        self.logger.info("Building Processor from config...")
        try:
            # Tokenizer
            tok_cfg = global_config.get('Tokenizer', {})
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tok_cfg['args']['tokenizer_path'],
                padding_side=tok_cfg['args']['padding_side'],
                truncation_side=tok_cfg['args']['truncation_side'],
                pad_token=tok_cfg['args']['pad_token'],
                bos_token=tok_cfg['args']['bos_token'],
                eos_token=tok_cfg['args']['eos_token'],
                unk_token=tok_cfg['args']['unk_token']
            )
            
            # Image Processor
            # Note: To enable augmentation, ensure 'aug_repeats' > 0 and 'do_augment': True
            # are present in the config['PreProcess']['ImageProcessor']['args'] section.
            pp_cfg = config.get('PreProcess', {})
            img_proc_cfg = pp_cfg.get('ImageProcessor', {})
            ImgProcCls = dynamic_import_class(img_proc_cfg['module'], img_proc_cfg['class'])
            self.image_processor = ImgProcCls(**img_proc_cfg.get('args', {}))

            # Processor (Combines ImageProcessor and Tokenizer)
            proc_cfg = pp_cfg.get('Processor', {})
            ProcCls = dynamic_import_class(proc_cfg['module'], proc_cfg['class'])
            self.processor = ProcCls(image_processor=self.image_processor, tokenizer=self.tokenizer)
            self.logger.info("Successfully built custom CMER Processor.")
        except Exception as e:
            self.logger.error(f"Failed to build processor: {e}")
            raise e

        # --- 3. Load Data List ---
        self.data_list = self._load_data_list()
        self.logger.info(f"Dataset loaded: {len(self.data_list)} samples.")
        
        # Initialize Collator
        self.collate_fn = CMERCollator(
            processor=self.processor, 
            max_length=self.max_length
        )
        self.need_reset = False 

    def _load_data_list(self):
        """Load label files, supporting jsonl and txt formats."""
        data_list = []
        for file_path in self.label_file_list:
            if not os.path.exists(file_path):
                self.logger.warning(f"Label file not found: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = parts[1]
                        data_list.append((img_path, label))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            img_path, text = self.data_list[idx]
            full_img_path = os.path.join(self.data_dir, img_path)
            
            # 1. Read original image
            image = Image.open(full_img_path).convert("RGB")
            
            # 2. Return raw objects directly
            # We do not call self.processor here because there is only one image, 
            # preventing batch padding at this stage.
            return {
                "image": image,  # Return PIL.Image object
                "text": text     # Return string
            }

        except Exception as e:
            self.logger.warning(f"Error loading sample {idx}: {e}. Replacing...")
            return self.__getitem__(np.random.randint(0, len(self)))

class CMERCollator:
    def __init__(self, processor, max_length=1024):
        """
        Args:
            processor: CMERProcessor instance
            max_length: Maximum text length (obtained from config)
        """
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        # Filter out None
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # 1. Split data
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # 2. Call Processor
        # Pass max_length here to suppress warnings
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,        # Image Batch Padding + Text Padding
            truncation=True,     # Enable truncation
            max_length=self.max_length  # <--- Critical fix: Specify max length
        )
        
        return inputs
import os
import glob
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from io import BytesIO
from PIL import Image
from functools import partial
from collections import Counter
from webdataset import handlers
import importlib
# Global counter for drop statistics, as used in the original code
_DROP_STATS = Counter()
def sanitize_keys(sample):
    new_sample = sample.copy() 
    for key in list(sample.keys()):
        if key.startswith("__"): 
            continue
        

        if "." in key:
            ext = key.split(".")[-1]

            if ext not in new_sample:
                new_sample[ext] = sample[key]
                
    return new_sample
# --- Helper Functions (Must be defined at module level for pickling in multiprocessing) ---

def keep_by_meta(sample, longside_max=12000, area_max=80_000_000, ar_max=20.0, 
                 shortside_min=16, require_positive_wh=True, max_tokens=1536, require_tokens=True):
    try:
        w = int(sample.get("width", 0) or 0)
        h = int(sample.get("height", 0) or 0)
        
        if require_positive_wh and (w <= 0 or h <= 0):
            _DROP_STATS["nonpos_wh"] += 1
            return False
        
        L, S = max(w, h), min(w, h)
        A = w * h
        ar = (w / h) if (h > 0) else math.inf
        
        if L > longside_max: _DROP_STATS["longside"] += 1; return False
        if A > area_max: _DROP_STATS["area"] += 1; return False
        if S < shortside_min: _DROP_STATS["shortside"] += 1; return False
        if (ar > ar_max) or (ar < 1.0 / ar_max): _DROP_STATS["ar"] += 1; return False
        
        tok = sample.get("tokens", None)
        if tok is None:
            if require_tokens: _DROP_STATS["no_tokens"] += 1; return False
        else:
            try:
                if int(tok) > max_tokens: _DROP_STATS["tokens"] += 1; return False
            except Exception:
                if require_tokens: _DROP_STATS["bad_tokens_val"] += 1; return False
        return True
    except Exception:
        _DROP_STATS["exception"] += 1
        return False

def parse_json_tuple_meta_only(sample):
    js, img_bytes = sample
    if isinstance(js, (bytes, bytearray)): js = json.loads(js.decode("utf-8"))
    elif isinstance(js, str): js = json.loads(js)
    
    return {
        "id": js.get('id', ''),
        "img_bytes": img_bytes,
        "tex": js["tex"],
        "tokens": int(js["tokens"]),
        "width": int(js.get("width", 0) or 0),
        "height": int(js.get("height", 0) or 0),
        "category": js.get('category', '')
    }

def add_ar_bin(sample, k=5, clamp=2.0):
    w, h = sample.get("width", 0), sample.get("height", 0)
    r = float(w) / float(h) if h else 1.0
    logar = math.log2(max(r, 1e-6))
    b = int(round(k * logar))
    sample["ar_bin"] = max(min(b, int(k * clamp)), -int(k * clamp))
    return sample

def add_short_edge_bin(sample, se_bin_size=96, se_min=96, se_max=1536):
    w, h = sample.get("width", 0), sample.get("height", 0)
    se = int(min(w, h)) if (w and h) else 0
    if se <= 0: se = se_min
    se = max(se_min, min(se, se_max))
    sample["se_bin"] = int(se // se_bin_size)
    return sample

def batch_to_inputs_decode_late_safe(batch, processor, max_length):
    images, texts, ids, categorys = [], [], [], []
    for s in batch:
        img_bytes = s.get("img_bytes", None)
        if not img_bytes:
            _DROP_STATS["empty_img_bytes"] += 1
            continue
        try:
            img = Image.open(BytesIO(img_bytes))
            img.load()
            img = img.convert("RGB")
        except Exception:
            _DROP_STATS["bad_img_bytes"] += 1
            continue
        ids.append(s.get('id'))
        images.append(img)
        texts.append(s["tex"])
        categorys.append(s.get('category'))
    
    if not images:
        raise handlers.SkipItem("all images in batch are broken/truncated")
        
    return processor(
        images=images,
        text=texts,
        ids=ids,
        categorys=categorys,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

@wds.pipelinefilter
def bucket_len_shortedge_ratio(data_iter, batch_size, pool_size=4096, len_key="tokens", 
                               se_key="se_bin", ar_key="ar_bin", len_bin_size=64, drop_last=True):
    pool = []
    def _drain_pool_pandas(pool):
        if not pool: return
        df = pd.DataFrame.from_records([{
            "idx": i,
            "len_bin": pool[i][len_key] // len_bin_size,
            "se_bin": pool[i][se_key],
            "ar_bin": pool[i][ar_key],
        } for i in range(len(pool))])
        
        all_leftovers = []
        for _, g in df.groupby(['len_bin', 'se_bin', 'ar_bin']):
            idxs = g["idx"].to_list()
            n_full = (len(idxs) // batch_size) * batch_size
            for i in range(0, n_full, batch_size):
                yield [pool[j] for j in idxs[i:i+batch_size]]
            if len(idxs) > n_full:
                all_leftovers.extend(idxs[n_full:])
        
        while len(all_leftovers) >= batch_size:
            yield [pool[j] for j in all_leftovers[:batch_size]]
            all_leftovers = all_leftovers[batch_size:]
        
        if (not drop_last) and all_leftovers:
            yield [pool[j] for j in all_leftovers]

    for sample in data_iter:
        pool.append(sample)
        if len(pool) >= pool_size:
            for batch in _drain_pool_pandas(pool): yield batch
            pool = []
    if pool:
        for batch in _drain_pool_pandas(pool): yield batch

# --- Main Dataset Class ---

class CMERWebDataSet(IterableDataset):
    def __init__(self, config, mode, logger, seed=None, epoch=None, task='rec'):
        super(CMERWebDataSet, self).__init__()
        
        # Config parsing
        global_config = config.get('Global', {})
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        
        self.mode = mode
        self.logger = logger
        self.data_dir = dataset_config['data_dir']
        self.batch_size = loader_config['batch_size_per_card']
        self.shuffle = loader_config.get('shuffle', False)
        
        # Specific CMER params
        self.max_length = dataset_config.get('max_length', 256)
        self.shuffle_buffer = dataset_config.get('shuffle_buffer', 10000)
        self.pool_size = dataset_config.get('pool_size', 8192)
        self.drop_last = dataset_config.get('drop_last', True if mode == 'train' else False)
        self.epochs = dataset_config.get('epochs', 1)
        
        processor_name = dataset_config.get('processor', 'CMERProcessor')
        module_path = dataset_config.get('processor_source', 'openrec.preprocess.cmer_label_encode')
        proc_module = importlib.import_module(module_path)
        ProcessorClass = getattr(proc_module, processor_name)
        proc_args = loader_config.get('processor_args', {})
        
        self.logger.info(f"Initializing {processor_name} with args: {proc_args}")
        self.processor = ProcessorClass(**proc_args)

        self.logger.info(f'Initialize CMER WebDataset: {self.data_dir} | Mode: {mode}')

    def _build_wds_pipeline(self, shards, epoch_idx=0):
        """Constructs the WebDataset pipeline."""
        if isinstance(shards, (list, tuple)):
            shard_list = shards
        else:
            shard_list = [shards]
            
        pipeline_stages = [
            wds.SimpleShardList(shard_list),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(self.shuffle_buffer if self.shuffle else 0),
            wds.map(sanitize_keys),
            wds.to_tuple("json", "jpg;png"),
            wds.map(parse_json_tuple_meta_only),
        ]

        # Filtering (Train only usually, but logic allows both)
        if self.shuffle_buffer != 0:
            pipeline_stages.append(
                wds.select(partial(
                    keep_by_meta,
                    longside_max=3840,
                    area_max=1536*1536,
                    ar_max=20.0,
                    shortside_min=0,
                    max_tokens=1536,
                    require_positive_wh=True,
                ))
            )

        # Bucketing and Batching
        pipeline_stages.extend([
            wds.map(add_ar_bin),
            wds.map(add_short_edge_bin),
            bucket_len_shortedge_ratio(
                batch_size=self.batch_size,
                pool_size=self.pool_size,
                len_key="tokens",
                se_key="se_bin",
                ar_key="ar_bin",
                len_bin_size=64,
                drop_last=self.drop_last,
            ),
            wds.map(partial(batch_to_inputs_decode_late_safe, 
                            processor=self.processor, 
                            max_length=self.max_length)),
        ])

        ds = wds.DataPipeline(*pipeline_stages)
        return ds.with_epoch(epoch_idx)

    def __iter__(self):
        """
        Iterates through the dataset. 
        Logic adapted from get_train_dataset and get_dataset.
        """
        all_datasets = []
        
        if self.mode == 'train':
            # Training logic: iterates through epoch folders
            epochs_to_use = list(range(0, self.epochs))
            for epoch_idx in epochs_to_use:
                # Assuming structure: root/epoch_0/*.tar
                epoch_path = os.path.join(self.data_dir, f"epoch_{epoch_idx}")
                train_shards = sorted(glob.glob(f"{epoch_path}/*.tar"))
                
                if not train_shards:
                    self.logger.warning(f"No .tar files found in {epoch_path}, skipping.")
                    continue
                
                ds = self._build_wds_pipeline(train_shards, epoch_idx)
                all_datasets.append(ds)
        else:
            # Eval/Test logic: iterates through package folders or flat structure
            # Logic adapted from get_dataset
            if os.path.exists(self.data_dir):
                # Check if it's a directory of packages or direct shards
                subdirs = [os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))]
                
                if subdirs:
                    # Package structure
                    for package_path in subdirs:
                        shards = sorted(glob.glob(f"{package_path}/*.tar"))
                        if shards:
                            ds = self._build_wds_pipeline(shards)
                            all_datasets.append(ds)
                else:
                    # Flat structure
                    shards = sorted(glob.glob(f"{self.data_dir}/*.tar"))
                    if shards:
                        ds = self._build_wds_pipeline(shards)
                        all_datasets.append(ds)

        if not all_datasets:
            raise RuntimeError(f"No data found in {self.data_dir}")

        # Chain the datasets (ChainIterDataset logic)
        for ds in all_datasets:
            for sample in ds:
                yield sample

    def __len__(self):
        # WebDataset length is often approximate or unknown until iteration
        # Returning a placeholder or calculating based on num_samples if available in metadata
        return 0
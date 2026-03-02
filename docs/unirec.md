# UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters

\[[Paper](https://arxiv.org/pdf/2512.21095)\] \[[ModelScope Model](https://www.modelscope.cn/models/topdktu/unirec-0.1b)\] \[[HuggingFace Model](https://huggingface.co/topdu/unirec-0.1b)\] \[[ModelScope Demo](https://www.modelscope.cn/studios/topdktu/OpenOCR-UniRec-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-UniRec-Demo)\] \[[Local Demo](#local-demo)\] \[[UniRec40M Dataset](https://huggingface.co/datasets/topdu/UniRec40M)\]

## Introduction

**UniRec-0.1B** is a unified recognition model with only 0.1B parameters, designed for high-accuracy and efficient recognition of plain text (words, lines, paragraphs), mathematical formulas (single-line, multi-line), and mixed content in both Chinese and English.

It addresses structural variability and semantic entanglement by using a hierarchical supervision training strategy and a semantic-decoupled tokenizer. Despite its small size, it achieves performance comparable to or better than much larger vision-language models.

## Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install openocr-python==0.1.5

# Or install from source
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
python build_package.py
pip install ./build/dist/openocr_python-*.whl
```

Recognize text, formulas, and tables using Vision-Language Model:

```bash
# Basic usage
openocr --task unirec --input_path path/to/img

# Process directory
openocr --task unirec --input_path ./images --output_path ./results
```

### Local Demo

Launch Gradio web interface for universal recognition:

```bash
pip install gradio
openocr --task launch_unirec_demo --server_port 7862 --share
```

### Python API Usage

```python
from openocr import OpenOCR

# Initialize UniRec
unirec = OpenOCR(task='unirec')

# Recognize text/formula/table
result_text, generated_ids = unirec(
    image_path='path/to/image.jpg',
    max_length=2048
)
print(f"Result: {result_text}")
```

## Get Started with Source

### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocr python==3.10
conda activate openocr
# install gpu version torch >=1.13.0
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# or cpu version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
git clone https://github.com/Topdu/OpenOCR.git
```

### Downloding the UniRec Model from ModelScope or Hugging Face

```shell
cd OpenOCR
pip install -r requirements.txt
# download model from modelscope
modelscope download topdktu/unirec-0.1b --local_dir ./unirec-0.1b
# or download model from huggingface
huggingface-cli download topdu/unirec-0.1b --local-dir ./unirec-0.1b
```

### Inference

```shell
python tools/infer_rec.py --c ./configs/rec/unirec/focalsvtr_ardecoder_unirec.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

### Finetuning on Custom Dataset

Additional dependencies:

```shell
pip install PyMuPDF
pip install pdf2image
pip install numpy==1.26.4
pip install albumentations==1.4.24
pip install transformers==4.49.0
pip install -U flash-attn --no-build-isolation
```

UniRec uses `NaSizeDataSet` for training, which groups images by similar dimensions for efficient batch processing with native image sizes.

**Directory Structure:**

```
your_dataset/
├── image_lmdb/           # LMDB containing images
│   ├── data.mdb
│   └── lock.mdb
└── label_key.json        # JSON file with labels grouped by image size
```

**Creating Custom DataSet:**

```python
import lmdb
import json
import math
from PIL import Image
import io

def create_nasize_dataset(image_label_pairs, output_lmdb_path, output_json_path, prefix='custom0001', divided_factor=64):
    """
    Create NaSizeDataSet format dataset.

    Args:
        image_label_pairs: List of (image_path, label) tuples
        output_lmdb_path: Path to output LMDB directory
        output_json_path: Path to output JSON file
        prefix: Prefix for file_name keys (used to identify which LMDB to read from)
        divided_factor: Image dimensions will be rounded up to multiples of this value
    """
    # Create LMDB
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)

    # Group by image size
    size_groups = {}

    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(image_label_pairs):
            # Read image
            with open(img_path, 'rb') as f:
                img_data = f.read()

            # Get image size and round up to multiples of divided_factor
            img = Image.open(io.BytesIO(img_data))
            w, h = img.size
            w_rounded = int(math.ceil(w / divided_factor) * divided_factor)
            h_rounded = int(math.ceil(h / divided_factor) * divided_factor)
            size_key = f"{w_rounded}_{h_rounded}"

            # Create unique file_name key with prefix
            # IMPORTANT: file_name must start with prefix for LMDB lookup
            file_name = f"{prefix}_{idx:06d}"

            # Store image in LMDB
            txn.put(file_name.encode('utf-8'), img_data)

            # Add to size group
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append({
                "file_name": file_name,
                "label": label
            })

    env.close()

    # Save JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(size_groups, f, ensure_ascii=False, indent=2)

    print(f"Created dataset with {sum(len(v) for v in size_groups.values())} samples")
    print(f"Size groups: {list(size_groups.keys())}")

# Example usage
image_label_pairs = [
    ('path/to/img1.jpg', 'Hello World'),
    ('path/to/img2.png', '\\( E = mc^2 \\)'),
    ('path/to/img3.jpg', '这是中文'),
]
create_nasize_dataset(image_label_pairs, './your_dataset/image_lmdb', './your_dataset/label_key.json', prefix='custom0001')
```

**JSON Label Format (`label_key.json`):**

The JSON file uses image dimensions as keys (format: `width_height`) and contains lists of sample information:

```json
{
    "640_64": [
        {
            "file_name": "prefix_sample001",
            "label": "Recognition text or formula"
        },
        {
            "file_name": "prefix_sample002",
            "label": "Another sample text"
        }
    ],
    "320_128": [
        {
            "file_name": "prefix_sample003",
            "label": "Text with different size"
        }
    ]
}
```

**Key Fields:**

- `{w}_{h}`: the size_key means Image size in pixels(width and height)
- `file_name`: Unique key to retrieve image from LMDB (must match the key in LMDB)
- `label`: Ground truth text. For formulas, use LaTeX format wrapped in `\( ... \)` (inline) or `\[ ... \]` (display)

#### Label Format Examples

| Content Type    | Label Format                                               |
| --------------- | ---------------------------------------------------------- |
| Plain text      | `Hello World`                                              |
| Inline formula  | `The formula \( x^2 + y^2 \) is important`                 |
| Display formula | `\[ \int_0^1 f(x) dx = F(1) - F(0) \]`                     |
| Mixed content   | `Given \( a > 0 \), we have \[ \sum_{n=1}^{\infty} a^n \]` |
| Chinese text    | `这是一段中文文本`                                         |

#### Finetuning Configuration

Copy the `configs/rec/unirec/focalsvtr_ardecoder_unirec.yml` file to `configs/rec/unirec/focalsvtr_ardecoder_unirec_finetune.yml` and modify the following fields:

```yaml

Global:
  ...
  output_dir: ./output/rec/unirec_finetune
  ...

...
Train:
  dataset:
    name: NaSizeDataSet
    divided_factor: &divided_factor [64, 64]  # w, h
    max_side: &max_side [960, 1408]  # w, h
    root_path: ./your_dataset  # Directory containing image_lmdb/ and label_key.json. If using UniRec40M, setting to path/to/UniRec40M
    custom_data: True  # Enable custom dataset mode
    add_return: True
    zoom_min_factor: 4
    use_zoom: True
    use_aug: True
    use_linedata: False # if True, use linedata in UniRec40M
    all_data: False # if True, use all UniRec40M
    # Optional: specify custom paths (defaults to root_path/image_lmdb and root_path/label_key.json)
    custom_lmdb_path: ['./your_dataset/image_lmdb', './your_dataset/image_lmdb']
    custom_label_json_path: ['./your_dataset/label_key.json', './your_dataset/label_key.json']
    custom_prefix: ['custom0001', 'custom0002']  # Must match the prefix used when creating the dataset
    custom_ratio_sample: [1, 1]  # Sampling ratio for custom data
    transforms:
      - UniRecLabelEncode:
          max_text_length: *max_text_length
          vlmocr: True
          tokenizer_path: *vlm_ocr_config
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  sampler:
    name: NaSizeSampler
    min_bs: 1
    max_bs: 24
  loader:
    shuffle: True
    batch_size_per_card: 16
    drop_last: True
    num_workers: 4

...
```

**Key Configuration Options:**

- `root_path`: Root directory containing your dataset
- `custom_data: True`: **Required** - Enable custom dataset mode for finetuning
- `custom_lmdb_path`: (Optional) Custom path to LMDB directory, defaults to `{root_path}/image_lmdb`. Can be a list for multiple LMDBs.
- `custom_label_json_path`: (Optional) Custom path to label JSON file, defaults to `{root_path}/label_key.json`. Can be a list for multiple JSON files.
- `custom_prefix`: (Optional) Prefix for file_name identification, defaults to `custom0001`. Must match the prefix used when creating the dataset. Can be a list corresponding to multiple LMDBs.
- `custom_ratio_sample`: (Optional) Sampling ratio for custom data, defaults to `1`. Can be a list for multiple datasets.
- `use_aug`: Enable/disable data augmentation
- `max_side`: Maximum image dimensions \[width, height\]
- `divided_factor`: Image dimensions will be rounded to multiples of these values

**Run Finetuning:**

```shell
cd OpenOCR
# download model from modelscope
modelscope download topdktu/unirec-0.1b --local_dir ./unirec-0.1b
# or download model from huggingface
huggingface-cli download topdu/unirec-0.1b --local-dir ./unirec-0.1b

# Single GPU
python tools/train_rec.py --c ./focalsvtr_ardecoder_unirec_finetune.yml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c ./focalsvtr_ardecoder_unirec_finetune.yml
```

### Training from Scratch

Additional dependencies:

```shell
pip install PyMuPDF
pip install pdf2image
pip install numpy==1.26.4
pip install albumentations==1.4.24
pip install transformers==4.49.0
pip install -U flash-attn --no-build-isolation
```

It is recommended to organize your working directory as follows:

```shell
|-UniRec40M    # Main directory for UniRec40M dataset
|-OpenOCR      # Directory for OpenOCR-related files
|-evaluation   # Directory for evaluation dataset
```

Download the UniRec40M dataset from Hugging Face

```shell
# downloading small data for quickly training
huggingface-cli download topdu/UniRec40M --include "hiertext_lmdb/**" --repo-type dataset --local-dir ./UniRec40M/
huggingface-cli download topdu/OpenOCR-Data --include "evaluation/**" --repo-type dataset --local-dir ./
```

Run the following command to train the model quickly:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=23333 --nproc_per_node=8 tools/train_rec.py --c configs/rec/unirec/focalsvtr_ardecoder_unirec.yml
```

Downloading the full dataset requires 3.5 TB of available storage space. Then, you need to merge the split files named `data.mdb.part_*` (located in `HWDB2Train`, `ch_pdf_lmdb`, and `en_pdf_lmdb`) into a single `data.mdb` file. Execute the commands below step by step:

```shell
# downloading full data
huggingface-cli download topdu/UniRec40M --repo-type dataset --local-dir ./UniRec40M/
cd UniRec40M/HWDB2Train/image_lmdb & cat data.mdb.part_* > data.mdb
cd UniRec40M/ch_pdf_lmdb & cat data.mdb.part_* > data.mdb
cd UniRec40M/en_pdf_lmdb & cat data.mdb.part_* > data.mdb
```

And modify the `configs/rec/unirec/focalsvtr_ardecoder_unirec.yml` file as follows:

```yaml
...
Train:
  dataset:
    name: NaSizeDataSet
    divided_factor: &divided_factor [64, 64] # w, h
    max_side: &max_side [960, 1408] # [64*30, 64*44] # w, h [960, 1408] #
    root_path: path/to/UniRec40M
    add_return: True
    zoom_min_factor: 4
    use_zoom: True
    all_data: True
    test_data: False
    use_aug: True
    use_linedata: True
    transforms:
      - UniRecLabelEncode: # Class handling label
          max_text_length: *max_text_length
          vlmocr: True
          tokenizer_path: *vlm_ocr_config # path to tokenizer, e.g. 'vocab.json', 'merges.txt'
      - KeepKeys:
          keep_keys: ['image', 'label', 'length'] # dataloader will return list in this order
  sampler:
    name: NaSizeSampler
    # divide_factor: to ensure the width and height dimensions can be devided by downsampling multiple
    min_bs: 1
    max_bs: 24
  loader:
    shuffle: True
    batch_size_per_card: 64
    drop_last: True
    num_workers: 8
...
```

## Citation

If you find our method useful for your research, please cite:

```bibtex
@article{du2025unirec,
  title={UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters},
  author={Yongkun Du and Zhineng Chen and Yazhen Xie and Weikang Bai and Hao Feng and Wei Shi and Yuchen Su and Can Huang and Yu-Gang Jiang},
  journal={arXiv preprint arXiv:2512.21095},
  year={2025}
}
```

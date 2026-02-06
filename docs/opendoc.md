# OpenDoc-0.1B: Ultra-Lightweight Document Parsing System with 0.1B Parameters

- \[[Quick Start](#quick-start)\] \[[ModelScope Demo](https://www.modelscope.cn/studios/topdktu/OpenDoc-0.1B-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenDoc-0.1B-Demo)\] \[[Local Demo](#local-demo)\]

## Introduction

**OpenDoc-0.1B** is an ultra-lightweight document parsing system featuring only 0.1 billion parameters. It operates through a sophisticated two-stage pipeline: first, it utilizes [PP-DocLayoutV2](https://www.paddleocr.ai/latest/version3.x/module_usage/layout_analysis.html) for precise layout analysis; second, it employs an enhanced, in-house [UniRec-0.1B](./unirec.md) model for the unified recognition of text, formulas, and tables. While the original version of UniRec-0.1B focused solely on text and formulas, this rebuilt iteration integrates comprehensive table recognition capabilities. Supporting both Chinese and English document parsing, **OpenDoc-0.1B** achieves an impressive **90.57%** score on [OmniDocBench (v1.5)](https://github.com/opendatalab/OmniDocBench/tree/main?tab=readme-ov-file#end-to-end-evaluation), demonstrating superior performance that outrivals many large-scale multimodal document parsing models.

## Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install openocr-python

# Or install from source
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
python build_package.py
pip install ./build/dist/openocr_python-*.whl
pip install onnxruntime
```

Parse documents with layout analysis, table/formula/table recognition:

```bash
# Full document parsing with all outputs
openocr --task doc --input_path path/to/img --use_layout_detection --save_vis --save_json --save_markdown

# Parse PDF document
openocr --task doc --input_path document.pdf --use_layout_detection --save_vis --save_json --save_markdown

# Custom layout threshold
openocr --task doc --input_path path/to/img --use_layout_detection --save_vis --save_json --save_markdown --layout_threshold 0.5
```

### Local Demo

Launch Gradio web interface for document parsing:

```bash
openocr --task launch_opendoc_demo --server_port 7862 --share
```

### Python API Usage

```python
from openocr import OpenOCR

# Initialize OpenDoc
doc_parser = OpenOCR(
    task='doc',
    use_layout_detection=True,
)

# Parse document
result = doc_parser(image_path='path/to/document.jpg')

# Save results
doc_parser.save_to_markdown(result, './output')
doc_parser.save_to_json(result, './output')
doc_parser.save_visualization(result, './output')
```

## Get Started with Source

### Requirements

```bash
conda create -n openocr python=3.10
conda activate openocr

git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
pip install -r requirements.txt

python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddlex
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.49.0
pip uninstall opencv-python
pip install pypdfium2
pip install opencv-contrib-python
```

### Download UniRec-0.1B model

```bash
# download model from modelscope
modelscope download topdktu/unirec-0.1b --local_dir ./unirec-0.1b
# or download model from huggingface
huggingface-cli download topdu/unirec-0.1b --local-dir ./unirec-0.1b
```

### Inference

```bash
# cpu
python tools/infer_doc.py --input_path ../doc_img_or_pdf --output_path ./output --gpus -1 --is_save_vis_img --is_save_json --is_save_markdown --pretty
# gpu
python tools/infer_doc.py --input_path ../doc_img_or_pdf --output_path ./output --gpus 0 --is_save_vis_img --is_save_json --is_save_markdown --pretty
# multi gpu
python tools/infer_doc.py --input_path ../doc_img_or_pdf --output_path ./output --gpus 0,1,2,3,4,5,6,7 --is_save_vis_img --is_save_json --is_save_markdown --pretty
```

### Reproduce the metrics on OmniDocBench

```
python tools/infer_doc.py --input_path ../doc_img_or_pdf --output_path ./output --gpus 0 --is_save_vis_img --is_save_json --is_save_markdown
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

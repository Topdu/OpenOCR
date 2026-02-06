# OpenOCR: An Open-Source Toolkit for General-OCR Research and Applications

## Recent Updates

- **0.1.0**: Use a unified interface for OCR, Document Parsing, and Unirec
- **0.0.10**: Remove OpenCV version restrictions.
- **0.0.9**: Fixing torch inference bug.
- **0.0.8**: Automatic Downloading ONNX model.
- **0.0.7**: Releasing the feature of [ONNX model export for wider compatibility](#export-onnx-model).

# Quick Start Guide

## Installation

```bash
# Install from PyPI (recommended)
pip install openocr-python

# Or install from source
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
python build_package.py
pip install ./build/dist/openocr_python-*.whl
pip install huggingface-hub==0.36.0 modelscope
```

## Command Line Usage

### 1. Text Detection + Recognition (OCR)

End-to-end OCR for Chinese/English text detection and recognition:

```bash
# Basic usage
openocr --task ocr --input_path path/to/img

# With visualization
openocr --task ocr --input_path path/to/img --is_vis

# Process directory with custom output
openocr --task ocr --input_path ./images --output_path ./results --is_vis

# Use server mode (higher accuracy)
openocr --task ocr --input_path path/to/img --mode server
```

### 2. Text Detection Only

Detect text regions without recognition:

```bash
# Basic detection
openocr --task det --input_path path/to/img

# With visualization
openocr --task det --input_path path/to/img --is_vis

# Use polygon detection (more accurate for curved text)
openocr --task det --input_path path/to/img --det_box_type poly
```

### 3. Text Recognition Only

Recognize text from cropped word/line images:

```bash
# Basic recognition
openocr --task rec --input_path path/to/img

# Use server mode (higher accuracy)
openocr --task rec --input_path path/to/img --mode server

# Batch processing
openocr --task rec --input_path ./word_images --rec_batch_num 16
```

### 4. Universal Recognition (UniRec)

Recognize text, formulas, and tables using Vision-Language Model:

```bash
# Basic usage
openocr --task unirec --input_path path/to/img

# Process directory
openocr --task unirec --input_path ./images --output_path ./results
```

### 5. Document Parsing (OpenDoc)

Parse documents with layout analysis, table/formula/table recognition:

```bash
# Full document parsing with all outputs
openocr --task doc --input_path path/to/img --use_layout_detection --save_vis --save_json --save_markdown

# Parse PDF document
openocr --task doc --input_path document.pdf --use_layout_detection --save_vis --save_json --save_markdown

# Custom layout threshold
openocr --task doc --input_path path/to/img --use_layout_detection --save_vis --save_json --save_markdown --layout_threshold 0.5
```

## Launch Interactive Demos

```bash
# Install gradio
pip install gradio==4.20.0
```

### OCR Demo

Launch Gradio web interface for OCR tasks:

```bash
pip install onnxruntime
# Local access only
openocr --task launch_openocr_demo --server_port 7860

# Public share link
openocr --task launch_openocr_demo --server_port 7860 --share
```

### UniRec Demo

Launch Gradio web interface for universal recognition:

```bash
pip install onnxruntime
openocr --task launch_unirec_demo --server_port 7861 --share
```

### OpenDoc Demo

Launch Gradio web interface for document parsing:

```bash
pip install onnxruntime
openocr --task launch_opendoc_demo --server_port 7862 --share
```

## Python API Usage

### OCR Task

```python
from openocr import OpenOCR

# Initialize OCR engine
ocr = OpenOCR(task='ocr', mode='mobile')

# Process single image
results, time_dicts = ocr(
    image_path='path/to/image.jpg',
    save_dir='./output',
    is_visualize=True
)

# Access results
for result in results:
    for line in result:
        print(f"Text: {line['text']}, Score: {line['score']}")
```

### Detection Task

```python
from openocr import OpenOCR

# Initialize detector
detector = OpenOCR(task='det')

# Detect text regions
results = detector(image_path='path/to/image.jpg')

# Access detection boxes
boxes = results[0]['boxes']
print(f"Found {len(boxes)} text regions")
```

### Recognition Task

```python
from openocr import OpenOCR

# Initialize recognizer
recognizer = OpenOCR(task='rec', mode='server')

# Recognize text
results = recognizer(image_path='path/to/word.jpg')

# Access recognition result
text = results[0]['text']
score = results[0]['score']
print(f"Text: {text}, Confidence: {score}")
```

### UniRec Task

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

### Document Parsing Task

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

## Common Parameters

- `--task`: Task type (ocr, det, rec, unirec, doc, launch\_\*\_demo)
- `--input_path`: Input image/PDF path or directory
- `--output_path`: Output directory (default: openocr_output/{task})
- `--use_gpu`: GPU usage (auto, true, false)
- `--mode`: Model mode (mobile, server) - server mode has higher accuracy
- `--is_vis`: Visualize results
- `--save_vis`: Save visualization (doc task)
- `--save_json`: Save JSON results (doc task)
- `--save_markdown`: Save Markdown results (doc task)

## Output Structure

Results are saved to `openocr_output/{task}/` by default:

- **OCR task**: `ocr_results.txt` + visualization images (if --is_vis)
- **Detection task**: `det_results.txt` + visualization images (if --is_vis)
- **Recognition task**: `rec_results.txt`
- **UniRec task**: `unirec_results.txt`
- **Doc task**: JSON files, Markdown files, visualization images (based on flags)

## For More Information

Visit: https://github.com/Topdu/OpenOCR

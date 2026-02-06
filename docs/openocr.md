# OpenOCR: A general OCR system with accuracy and efficiency

⚡\[[Quick Start](#quick-start)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[Local Demo](#local-demo)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]

We proposed strategies to comprehensively enhance CTC-based STR models and developed a novel CTC-based method, [SVTRv2](../configs/rec/svtrv2/). SVTRv2 can outperform previous attention-based STR methods in terms of accuracy while maintaining the advantages of CTC, such as fast inference and robust recognition of long text. These features make SVTRv2 particularly well-suited for practical applications. To this end, building on SVTRv2, we develop a practical version of the model from scratch on publicly available Chinese and English datasets. Combined with a detection model, this forms a general OCR system with accuracy and efficiency, **OpenOCR**. Comparing with [PP-OCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html) baseline in the [OCR competition leaderboard](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard), OpenOCR (mobile) achieve a 4.5% improvement in terms of accuracy, while preserving quite similar inference speed on NVIDIA 1080Ti GPU.

| Model               | Config                                                                              | E2E Metric | Downloading                                                                              |
| ------------------- | ----------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| PP-OCRv4            |                                                                                     | 62.77%     | [PaddleOCR Model List](../../ppocr/model_list.md)                                        |
| SVTRv2 (Rec Server) | [configs/rec/svtrv2/svtrv2_ch.yml](../configs/rec/svtrv2/svtrv2_ch.yml)             | 68.81%     | [Google Dirve](https://drive.google.com/file/d/13LXbIVEyx2Aat3X_vVte4JQgQ7yJWdxH/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth) |
| RepSVTR (Mobile)    | [Rec: configs/rec/svtrv2/repsvtr_ch.yml](../configs/rec/svtrv2/repsvtr_ch.yml) <br> [Det: configs/det/dbnet/repvit_db.yml](../configs/det/dbnet/repvit_db.yml) | 67.22%     | [Rec: Google Drive](https://drive.google.com/file/d/1DNfarP_UmTqZnENjmmQHCexqzVmrIfLF/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth) <br>  [Det: Google Drive](https://drive.google.com/file/d/1eR6k5NitCvFEiGlYx1lAArVupIszfEmM/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth) |

## Quick Start

**Note**: OpenOCR supports inference using both the ONNX and Torch frameworks, with the dependency environments for the two frameworks being isolated. When using ONNX for inference, there is no need to install Torch, and vice versa.

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

#### 1. Text Detection + Recognition (OCR)

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

#### 2. Text Detection Only

Detect text regions without recognition:

```bash
# Basic detection
openocr --task det --input_path path/to/img

# With visualization
openocr --task det --input_path path/to/img --is_vis

# Use polygon detection (more accurate for curved text)
openocr --task det --input_path path/to/img --det_box_type poly
```

#### 3. Text Recognition Only

Recognize text from cropped word/line images:

```bash
# Basic recognition
openocr --task rec --input_path path/to/img

# Use server mode (higher accuracy)
openocr --task rec --input_path path/to/img --mode server

# Batch processing
openocr --task rec --input_path ./word_images --rec_batch_num 16
```

### Local Demo

Launch Gradio web interface for OCR tasks:

```bash
openocr --task launch_openoce_demo --server_port 7862 --share
```

### Python API Usage

#### 1. OCR Task

```python
from openocr import OpenOCR

# Initialize OCR engine
ocr = OpenOCR(mode='mobile', backend=='onnx')

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

#### 2. Detection Task

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

#### 3. Recognition Task

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

## Get Started with Source

#### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocr python==3.8
conda activate openocr
# install gpu version torch
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# or cpu version
conda install pytorch torchvision torchaudio cpuonly -c pytorch

git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
pip install -r requirements.txt
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth
# Rec Server model
# wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth
```

**Usage**:

```shell
# OpenOCR system: Det + Rec model
python tools/infer_e2e.py --img_path=/path/img_fold or /path/img_file
# Det model
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.infer_img=/path/img_fold or /path/img_file
# Rec model
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

## Fine-tuning on a Custom dataset

Referring to [Finetuning Det](./finetune_det.md) and [Finetuning Rec](./finetune_rec.md).

## Exporting to ONNX Engine

### Export ONNX model

```shell
pip install onnx
python tools/toonnx.py --c configs/rec/svtrv2/repsvtr_ch.yml --o Global.device=cpu
python tools/toonnx.py --c configs/det/dbnet/repvit_db.yml --o Global.device=cpu
```

The det onnx model is saved in `./output/det_repsvtr_db/export_det/det_model.onnx`.
The rec onnx model is saved in `./output/rec/repsvtr_ch/export_rec/rec_model.onnx`.

### Inference with ONNXRuntime

```shell
pip install onnxruntime
# OpenOCR system: Det + Rec model
python tools/infer_e2e.py --img_path=/path/img_fold or /path/img_file --backend=onnx --device=cpu --onnx_det_model_path=./output/det_repsvtr_db/export_det/det_model.onnx --onnx_rec_model_path=output/rec/repsvtr_ch/export_rec/rec_model.onnx
# Det model
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.backend=onnx Global.device=cpu  Global.infer_img=/path/img_fold or /path/img_file Global.onnx_model_path=./output/det_repsvtr_db/export_det/det_model.onnx
# Rec model
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_fold or /path/img_file Global.onnx_model_path=./output/rec/repsvtr_ch/export_rec/rec_model.onnx
```

## Results Showcase

### Detection results

<div align="center">
<img src=https://github.com/user-attachments/assets/4cf61a6d-c64f-4516-899b-5a7bb0c5314b height=300 />
</div>

### Recognition results

<div align="center">
<img src=https://github.com/user-attachments/assets/38992055-7f47-4865-bc5e-ced114e96c54 height=400 />
</div>

### Det + Rec System results

<div align="center">
<img src=https://github.com/user-attachments/assets/4df4bed2-180f-43bd-8ed6-25baf53edebd height=550 />
</div>
<div align="center">
<img src=https://github.com/user-attachments/assets/f8d7acf3-052f-4047-885e-26a90935394d height=500 />
</div>
<div align="center">
<img src=https://github.com/user-attachments/assets/40a0e071-1e03-45bb-b087-67a0960a26bf height=550 />
</div>

### **Detection Model Performance**

In the examples provided, OpenOCR's detection model generates bounding boxes that are generally more comprehensive and better aligned with the boundaries of text instances compared to PP-OCRv4. In addition, OpenOCR excels in distinguishing separate text instances, avoiding errors such as merging two distinct text instances into one or splitting a single instance into multiple parts. This indicates superior handling of **semantic completeness and spatial understanding**, making it particularly effective for complex layouts.

### **Recognition Model Generalization**

OpenOCR's recognition model demonstrates enhanced generalization capabilities when compared to PP-OCRv4. It performs exceptionally well in recognizing text under difficult conditions, such as:

- Artistic or stylized fonts.
- Handwritten text.
- Blurry or low-resolution images.
- Incomplete or occluded text.

Remarkably, the **OpenOCR mobile recognition model** delivers results comparable to the larger and more resource-intensive **PP-OCRv4 server model**. This highlights OpenOCR's efficiency and accuracy, making it a versatile solution across different hardware platforms.

### **System used in Real-World Scenarios**

As shown in Det + Rec System results, OpenOCR demonstrates outstanding performance in practical scenarios, including documents, tables, invoices, and similar contexts. This underscores its potential as a **general-purpose OCR system**. It is capable of adapting to diverse use cases with high accuracy and reliability.

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@inproceedings{Du2025SVTRv2,
      title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
      author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
      booktitle={ICCV},
      year={2025},
      pages={20147-20156}
}
```

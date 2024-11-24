# OpenOCR: A general OCR system with accuracy and efficiency

We proposed strategies to comprehensively enhance CTC-based STR models and developed a novel CTC-based method, [SVTRv2](../configs/rec/svtrv2/). SVTRv2 can outperform previous attention-based STR methods in terms of accuracy while maintaining the advantages of CTC, such as fast inference and robust recognition of long text sequences. These features make SVTRv2 particularly well-suited for commercial applications. To this end, building on SVTRv2, we develop a practical version of the model from scratch on publicly available Chinese and English datasets. Combined with a detection model, this forms an accurate and efficient general OCR system, OpenOCR. Comparing with [PP-OCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html) released by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), OpenOCR achieve a 4.5% improvement on the [OCR competition leaderboard](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard).

| Model               | Config                                                                              | E2E Metric | Downloading                                                                              |
| ------------------- | ----------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| PP-OCRv4            |                                                                                     | 62.77%     | [PaddleOCR Model List](../../ppocr/model_list.md)                                        |
| SVTRv2 (Rec Server) | [configs/rec/svtrv2/svtrv2_ch.yml](../configs/rec/svtrv2/svtrv2_ch.yml)             | 68.81%     | [Google Dirve](https://drive.google.com/file/d/13LXbIVEyx2Aat3X_vVte4JQgQ7yJWdxH/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth) |
| RepSVTR (Mobile)    | [Rec: configs/rec/svtrv2/repsvtr_ch.yml](../configs/rec/svtrv2/repsvtr_ch.yml) <br> [Det: configs/det/dbnet/repvit_db.yml](../configs/det/dbnet/repvit_db.yml) | 67.22%     | [Rec: Google Drive](https://drive.google.com/file/d/1DNfarP_UmTqZnENjmmQHCexqzVmrIfLF/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth) <br>  [Det: Google Drive](https://drive.google.com/file/d/1eR6k5NitCvFEiGlYx1lAArVupIszfEmM/view?usp=drive_link), [Github Released](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth) |

## Quick Start

#### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocre2e python==3.8
conda activate openocre2e
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

After installing dependencies, the following two installation methods are available. Either one can be chosen.

#### 1. Python Modules

```shell
pip install openocr-python
```

**Usage**:

```python
from openocr import OpenOCR

engine = OpenOCR()

img_path = '/path/img_path or /path/img_file'
result, elapse = engine(img_path)
print(result)
print(elapse)

# Server mode
engine = OpenOCR(mode='server')
```

#### 2. Clone this repository:

```shell
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
pip install -r requirements.txt
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

## Demo

```shell
pip install gradio==4.20.0

# start demo
python demo_gradio.py
```

## Fine-tuning on a Custom dataset

TODO

## Exporting to ONNX Engine

TODO

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
<img src=https://github.com/user-attachments/assets/4ad66441-28ad-4081-85b3-45532ccabf19 height=600 />
</div>
<div align="center">
<img src=https://github.com/user-attachments/assets/ecdcd7a7-2c62-490d-847f-9d3d64def3ee height=400 />
</div>
<div align="center">
<img src=https://github.com/user-attachments/assets/1fdfb5bc-6eb7-4a83-b212-45bfbab1fc74 height=500 />
</div>

### **Detection Model Performance**

In the examples provided in Detection results, OpenOCR's detection model generates bounding boxes that are more comprehensive and better aligned with the boundaries of text instances compared to PP-OCRv4. In addition, OpenOCR excels in distinguishing separate text instances, avoiding errors such as merging two distinct text instances into one or splitting a single instance into multiple parts. This indicates superior handling of **semantic completeness and spatial understanding**, making it particularly effective for complex layouts.

### **Recognition Model Generalization**

OpenOCR's recognition model demonstrates enhanced generalization capabilities when compared to PP-OCRv4. It performs exceptionally well in recognizing text under difficult conditions, such as:

- Artistic or stylized fonts.
- Handwritten text.
- Blurry or low-resolution images.
- Text with incomplete or occluded portions.

Remarkably, the **OpenOCR mobile recognition model** delivers results comparable to the larger and more resource-intensive **PP-OCRv4 server model**. This highlights OpenOCR's efficiency and optimization, making it a versatile solution across different hardware platforms.

### **System used in Real-World Scenarios**

Finally, as shown in Det + Rec System results, OpenOCR demonstrates outstanding performance in practical scenarios, including documents, tables, invoices, and similar contexts. This underscores its potential as a **general-purpose OCR system**, capable of adapting to diverse use cases with high accuracy and reliability.

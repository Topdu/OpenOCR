# OpenOCR: A general OCR system with accuracy and efficiency

We proposed strategies to comprehensively enhance CTC-based STR models and developed a novel CTC-based method, [SVTRv2](../configs/rec/svtrv2/). SVTRv2 can outperform previous attention-based STR methods in terms of accuracy while maintaining the advantages of CTC, such as fast inference and robust recognition of long text sequences. These features make SVTRv2 particularly well-suited for commercial applications. To this end, building on SVTRv2, we develop a practical version of the model from scratch on publicly available Chinese and English datasets. Combined with a detection model, this forms an accurate and efficient general OCR system, OpenOCR. Comparing with PP-OCRv4 released by PaddleOCR, OpenOCR achieve a 4.5% improvement on the [OCR competition leaderboard](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard).

| Model               | Config                                                                              | E2E Metric | Downloading                                                                              |
| ------------------- | ----------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| PP-OCRv4            |                                                                                     | 62.77%     | [PaddleOCR Model List](../../ppocr/model_list.md)                                        |
| SVTRv2 (Rec Server) | [configs/rec/svtrv2/svtrv2_ch.yml](../configs/rec/svtrv2/svtrv2_ch.yml)             | 68.81%     | [Google Dirve ](https://drive.google.com/file/d/13LXbIVEyx2Aat3X_vVte4JQgQ7yJWdxH/view?usp=drive_link) |
| RepSVTR (Mobile)    | [Rec: configs/rec/svtrv2/repsvtr_ch.yml](../configs/rec/svtrv2/repsvtr_ch.yml) <br> [Det: configs/det/dbnet/repvit_db.yml](../configs/det/dbnet/repvit_db.yml) | 67.22%     | [Rec: Google Drive](https://drive.google.com/file/d/1DNfarP_UmTqZnENjmmQHCexqzVmrIfLF/view?usp=drive_link) <br>  [Det: Google Drive](https://drive.google.com/file/d/1eR6k5NitCvFEiGlYx1lAArVupIszfEmM/view?usp=drive_link) |

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

The results show that OpenOCR’s detection model outperforms PP-OCRv4 in generating more complete and accurate text boundaries, effectively capturing entire text instances. This reflects its larger receptive field and its ability to avoid common issues like merging separate text instances or splitting a single instance into multiple fragments.

In terms of recognition, OpenOCR demonstrates superior adaptability to challenging scenarios, such as artistic fonts, handwriting, blur, low resolution, incomplete text, and occlusion. Notably, the OpenOCR mobile model performs at a level comparable to PP-OCRv4's larger server-side model, showcasing its efficiency and robustness.

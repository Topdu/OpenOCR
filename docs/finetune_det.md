# Fine-tuning Text Detection Model of OpenOCR System

1. [Data and Weights Preparation](#1-data-and-weights-preparation)
   - [1.1 Data Preparation](#11-data-preparation)
   - [1.2 Download Pre-trained Model](#12-download-pre-trained-model)
2. [Training](#2-training)
   - [2.1 Start Training](#21-start-training)
   - [2.2 Load Trained Model and Continue Training](#22-load-trained-model-and-continue-training)
3. [Evaluation and Test](#3-evaluation-and-test)
   - [3.1 Evaluation](#31-evaluation)
   - [3.2 Test](#32-test)
4. [ONNX Inference](#4-onnx-inference)

______________________________________________________________________

## Installation

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
```

#### Clone this repository:

```shell
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
pip install albumentations
pip install -r requirements.txt
```

This section uses the icdar2015 dataset as an example to introduce the training, evaluation, and testing of the detection model in OpenOCR.

## 1. Data and Weights Preparation

### 1.1 Data Preparation

**Note:** If you want to use your own dataset, please following the format of [icdar2015 dataset](https://aistudio.baidu.com/datasetdetail/46088).

Downloading datasets from [icdar2015 dataset](https://aistudio.baidu.com/datasetdetail/46088).

#### File Directory

```
OpenOCR/
icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         Training data of the icdar dataset
  └─ ch4_test_images/             Testing data of the icdar dataset
  └─ train_icdar2015_label.txt    Training annotations of the icdar dataset
  └─ test_icdar2015_label.txt     Testing annotations of the icdar dataset
```

The provided annotation file format is as follows, where the fields are separated by "\\t":

```
"Image file name                   json.dumps encoded image annotation information"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]], ...}]
```

Before being encoded with `json.dumps`, the image annotation information is a list containing multiple dictionaries. In each dictionary, the field `points` represents the coordinates (x, y) of the four corners of the text bounding box, arranged in a clockwise order starting from the top-left corner. The field `transcription` indicates the text content within the current bounding box.

To modify the training and evaluation dataset paths in the configuration file `./configs/det/dbnet/repvit_db.yml` to your own dataset paths, for example:

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ../icdar2015/text_localization/  # Root directory of the training dataset
    label_file_list: ["../icdar2015/text_localization/train_icdar2015_label.txt"]  # Path to the training label file
    ......
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ../icdar2015/text_localization/  # Root directory of the evaluation dataset
    label_file_list: ["../icdar2015/text_localization/test_icdar2015_label.txt"]  # Path to the evaluation label file
```

### 1.2 Download Pre-trained Model

First download the pre-trained model.

```bash
cd OpenOCR/
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth
```

______________________________________________________________________

## 2. Training

### 2.1 Start Training

```bash
# multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_det.py --c configs/det/dbnet/repvit_db.yml --o Global.pretrained_model=./openocr_det_repvit_ch.pth
# single GPU training
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 tools/train_det.py --c configs/det/dbnet/repvit_db.yml --o Global.pretrained_model=./openocr_det_repvit_ch.pth
```

### 2.2 Load Trained Model and Continue Training

If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_det.py --c configs/det/dbnet/repvit_db.yml --o Global.checkpoints=./your/trained/model
```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrained_model`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrained_model` will be loaded.

______________________________________________________________________

## 3. Evaluation and Test

### 3.1 Evaluation

OpenOCR calculates three indicators for evaluating performance of OCR detection task: Precision, Recall, and Hmean(F-Score).

```bash
python tools/eval_det.py --c configs/det/dbnet/repvit_db.yml --o Global.pretrained_model="{path/to/weights}/best.pth"
```

### 3.2 Test

Test the detection result on all images in the folder or a single image:

```bash
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.infer_img=/path/img_fold or /path/img_file Global.pretrained_model={path/to/weights}/best.pth
```

______________________________________________________________________

## 4. ONNX Inference

Firstly, we can convert Detection model to onnx model:

```bash
pip install onnx
python tools/toonnx.py --c ./configs/det/dbnet/repvit_db.yml --o Global.device=cpu Global.pretrained_model={path/to/weights}/best.pth
```

The onnx model is saved in `./output/det_repsvtr_db/export_det/det_model.onnx`.

The detection onnx model inference:

```bash
pip install onnxruntime
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_fold or /path/img_file Global.onnx_model_path=./output/det_repsvtr_db/export_det/det_model.onnx
```

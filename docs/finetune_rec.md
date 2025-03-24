# Fine-tuning Text Recognition Model of OpenOCR system

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

**Note:** If you want to use your own dataset, please following the following data format.

This section uses the icdar2015 recognition dataset as an example to introduce the training, evaluation, and testing of the recognition model in OpenOCR.

## 1. Data and Weights Preparation

### 1.1 Data Preparation

Downloading datasets from [icdar2015 recognition dataset](https://aistudio.baidu.com/datasetdetail/75418).

#### File Directory

```
OpenOCR/
ic15_data/
  └─ test/         Training data of the icdar dataset
  └─ train/             Testing data of the icdar dataset
  └─ rec_gt_test.txt    Training annotations of the icdar dataset
  └─ rec_gt_train.txt     Testing annotations of the icdar dataset
```

The provided annotation file format is as follows, where the fields are separated by "\\t":

```
"Image file name                   label"
test/word_2077.png    Underpass
```

### 1.2 Download Pre-trained Model

First download the pre-trained model.

```bash
cd OpenOCR/
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth
# Rec Server model
# wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth
```

## 2. Training

### 2.1 Start Training

```bash
# multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml
# single GPU training
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 tools/train_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml
```

### 2.2 Load Trained Model and Continue Training

If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.checkpoints=./your/trained/model
```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrained_model`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrained_model` will be loaded.

## 3. Evaluation and Test

### 3.1 Evaluation

OpenOCR calculates three indicators for evaluating performance of OCR recognition task: Precision, Recall, and Hmean(F-Score).

```bash
python tools/eval_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.pretrained_model="{path/to/weights}/best.pth"
```

### 3.2 Test

Test the recognition result on all images in the folder or a single image:

```bash
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.infer_img=/path/img_fold or /path/img_file Global.pretrained_model={path/to/weights}/best.pth
```

## 4. ONNX Inference

Firstly, we can convert recognition model to onnx model:

```bash
pip install onnx
python tools/toonnx.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.device=cpu Global.pretrained_model={path/to/weights}/best.pth
```

The onnx model is saved in `./output/rec/repsvtr_ch/export_rec/rec_model.onnx`.

The recognition onnx model infernce:

```bash
pip install onnxruntime
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_fold or /path/img_file Global.onnx_model_path=./output/rec/repsvtr_ch/export_rec/rec_model.onnx
```

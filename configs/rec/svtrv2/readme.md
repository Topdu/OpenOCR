# SVTRv2

- [SVTRv2](#svtrv2)
  - [1. Introduction](#1-introduction)
    - [1.1 Models and Results](#11-models-and-results)
  - [2. Environment](#2-environment)
  - [3. Model Training / Evaluation](#3-model-training--evaluation)
    - [Dataset Preparation](#dataset-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
    - [Latency Measurement](#latency-measurement)
  - [Citation](#citation)

<a name="1"></a>

## 1. Introduction

Paper:

> [SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](https://arxiv.org/abs/2411.15858)
> Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang

<a name="model"></a>
Connectionist temporal classification (CTC)-based scene text recognition (STR) methods, e.g., SVTR, are widely employed in OCR applications, mainly due to their simple architecture, which only contains a visual model and a CTC-aligned linear classifier, and therefore fast inference. However, they generally have worse accuracy than encoder-decoder-based methods (EDTRs), particularly in challenging scenarios. In this paper, we propose SVTRv2, a CTC model that beats leading EDTRs in both accuracy and inference speed. SVTRv2 introduces novel upgrades to handle text irregularity and utilize linguistic context, which endows it with the capability to deal with challenging and diverse text instances. First, a multi-size resizing (MSR) strategy is proposed to adaptively resize the text and maintain its readability. Meanwhile, we introduce a feature rearrangement module (FRM) to ensure that visual features accommodate the alignment requirement of CTC well, thus alleviating the alignment puzzle. Second, we propose a semantic guidance module (SGM). It integrates linguistic context into the visual model, allowing it to leverage language information for improved accuracy. Moreover, SGM can be omitted at the inference stage and would not increase the inference cost. We evaluate SVTRv2 in both standard and recent challenging benchmarks, where SVTRv2 is fairly compared with 24 mainstream STR models across multiple scenarios, including different types of text irregularity, languages, and long text. The results indicate that SVTRv2 surpasses all the EDTRs across the scenarios in terms of accuracy and speed.

### 1.1 Models and Results

The accuracy (%) and model files of SVTRv2 on the public dataset of scene text recognition are as follows:

Download all Configs, Models, and Logs from [Google Drive](https://drive.google.com/drive/folders/1i2EZVT-oxfDIDdhwQRm9E6Fk8s6qD3C1?usp=sharing).

|  Model   | Model size | Latency |
| :------: | :--------: | :-----: |
| SVTRv2-T |    5.13    |   5.0   |
| SVTRv2-S |   11.25    |   5.3   |
| SVTRv2-B |   19.76    |   7.0   |

- Test on Common Benchmarks from [PARSeq](https://github.com/baudm/parseq):

|  Model   |                        Training Data                         | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                             Config&Model&Log                              |
| :------: | :----------------------------------------------------------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :-----------------------------------------------------------------------: |
| SVTRv2-B |                  Synthetic datasets (MJ+ST)                  |     97.7     | 94.0 |      97.3       |     88.1      | 91.2 |  95.8  | 94.02 |                                   TODO                                    |
| SVTRv2-T | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) |     98.6     | 96.6 |      98.0       |     88.4      | 90.5 |  96.5  | 94.78 | [Google drive](https://drive.google.com/drive/folders/12ZUGkCS7tEhFhWa2RKKtyB0tPjhH4d9s?usp=drive_link) |
| SVTRv2-S | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) |     99.0     | 98.3 |      98.5       |     89.5      | 92.9 |  98.6  | 96.13 | [Google drive](https://drive.google.com/drive/folders/1mOG3EUAOsmD16B-VIelVDYf_O64q0G3M?usp=drive_link) |
| SVTRv2-B | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) |     99.2     | 98.0 |      98.7       |     91.1      | 93.5 |  99.0  | 96.57 | [Google drive](https://drive.google.com/drive/folders/11u11ptDzQ4BF9RRsOYdZnXl6ell2h4jN?usp=drive_link) |

- Test on Union14M-L benchmark from [Union14M](https://github.com/Mountchicken/Union14M/).

|  Model   |                         Traing Data                          | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :------: | :----------------------------------------------------------: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
| SVTRv2-B |                  Synthetic datasets (MJ+ST)                  | 74.6  |        25.2         |   57.6   |    69.7     |  77.9   |      68.0       |  66.9   | 62.83 | Same as the above table |
| SVTRv2-T | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 83.6  |        76.0         |   71.2   |    82.4     |  77.2   |      82.3       |  80.7   | 79.05 | Same as the above table |
| SVTRv2-S | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 88.3  |        84.6         |   76.5   |    84.3     |  83.3   |      85.4       |  83.5   | 83.70 | Same as the above table |
| SVTRv2-B | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 90.6  |        89.0         |   79.3   |    86.1     |  86.2   |      86.7       |  85.1   | 86.14 | Same as the above table |

- Test on [LTB](../smtr/readme.md) and [OST](https://github.com/wangyuxin87/VisionLAN).

|  Model   |                         Traing Data                          |  LTB  | OST  |    Config&Model&Log     |
| :------: | :----------------------------------------------------------: | :---: | :--: | :---------------------: |
| SVTRv2-T | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 47.83 | 71.4 | Same as the above table |
| SVTRv2-S | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 47.57 | 78.0 | Same as the above table |
| SVTRv2-B | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 50.23 | 80.0 | Same as the above table |

- Training and test on Chinese dataset, from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|  Model   | Scene | Web  | Document | Handwriting |  Avg  |                                            Config&Model&Log                                             |
| :------: | :---: | :--: | :------: | :---------: | :---: | :-----------------------------------------------------------------------------------------------------: |
| SVTRv2-T | 77.8  | 78.8 |   99.3   |    62.0     | 79.45 | [Google drive](https://drive.google.com/drive/folders/1vqTFonJV83SXVFrGhL31zXq7aOLwjnGD?usp=drive_link) |
| SVTRv2-S | 81.1  | 81.2 |   99.3   |    65.0     | 81.64 | [Google drive](https://drive.google.com/drive/folders/1X3hqArfvRIRtuYLHDtSQheQmDc_oXpY6?usp=drive_link) |
| SVTRv2-B | 83.5  | 83.3 |   99.5   |    67.0     | 83.31 | [Google drive](https://drive.google.com/drive/folders/1ZDECKXf8zZFhcKKKpvicg43Ho85uDZkF?usp=drive_link) |

<a name="2"></a>

## 2. Environment

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
git clone -b develop https://github.com/Topdu/OpenOCR.git
cd OpenOCR
# Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.8
conda activate openocr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

<a name="3"></a>

## 3. Model Training / Evaluation

### Dataset Preparation

Referring to [Downloading Datasets](../../../docs/svtrv2.md#downloading-datasets)

### Training

```shell
# First stage
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_rctc.yml

# Second stage
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port=23332 --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml --o Global.pretrained_model=./output/rec/u14m_filter/svtrv2_rctc/best.pth

# For Multi RTX 4090
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=23333 --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_rctc.yml
# 20epoch runs for about 6 hours
```

### Evaluation

```shell
# short text: Common, Union14M-Benchmark, OST
python tools/eval_rec_all_en.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml

# long text: LTB
python tools/eval_rec_all_long.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml --o Eval.loader.max_ratio=20
```

After a successful run, the results are saved in a csv file in `output_dir` in the config file.

### Inference

```shell
python tools/infer_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

### Latency Measurement

Firstly, downloading the IIIT5K images from [Google Drive](https://drive.google.com/drive/folders/1Po1LSBQb87DxGJuAgLNxhsJ-pdXxpIfS?usp=drive_link). Then, running the following command:

```shell
python tools/infer_rec.py --c configs/rec/SVTRv2/svtrv2_smtr_gtc_rctc_infer.yml --o Global.infer_img=../iiit5k_test_image
```

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@inproceedings{Du2024SVTRv2,
      title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
      author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
      booktitle={ICCV},
      year={2025}
}
```

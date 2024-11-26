# SMTR

- [SMTR](#smtr)
  - [1. Introduction](#1-introduction)
  - [2. Environment](#2-environment)
    - [Dataset Preparation](#dataset-preparation)
  - [3. Model Training / Evaluation](#3-model-training--evaluation)
  - [Citation](#citation)

<a name="1"></a>

## 1. Introduction

Paper:

> [Out of Length Text Recognition with Sub-String Matching](https://arxiv.org/abs/2407.12317)
> Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang

<a name="model"></a>
Scene Text Recognition (STR) methods have demonstrated robust performance in word-level text recognition. However, in applications the text image is sometimes long due to detected with multiple horizontal words. It triggers the requirement to build long text recognition models from readily available short word-level text datasets, which has been less studied previously. In this paper, we term this the Out of Length (OOL) text recognition. We establish the first Long Text Benchmark (LTB) to facilitate the assessment of different methods in long text recognition. Meanwhile, we propose a novel method called OOL Text Recognition with sub-String Matching (SMTR). SMTR comprises two cross-attention-based modules: one encodes a sub-string containing multiple characters into next and previous queries, and the other employs the queries to attend to the image features, matching the sub-string and simultaneously recognizing its next and previous character. SMTR can recognize text of arbitrary length by iterating the process above. To avoid being trapped in recognizing highly similar sub-strings, we introduce a regularization training to compel SMTR to effectively discover subtle differences between similar sub-strings for precise matching. In addition, we propose an inference augmentation to alleviate confusion caused by identical sub-strings and improve the overall recognition efficiency. Extensive experimental results reveal that SMTR, even when trained exclusively on short text, outperforms existing methods in public short text benchmarks and exhibits a clear advantage on LTB.

The accuracy (%) and model files of SMTR on the public dataset of scene text recognition are as follows:

- Syn: Synth dataset(MJ+ST) from [PARSeq](https://github.com/baudm/parseq)

- U14M: Union14M-L from [Union14M](https://github.com/Mountchicken/Union14M/)

- Test on Long Text Benchmark ([Download LTB](https://drive.google.com/drive/folders/1NChdlw7ustbXtlFBmh_0xnHvRkffb9Ge?usp=sharing)):

|   Model   | Training Data | LTB  |                                        Config&Model&Log                                         |
| :-------: | :-----------: | :--: | :---------------------------------------------------------------------------------------------: |
|   SMTR    |      Syn      | 39.6 |  [link](https://drive.google.com/drive/folders/11SplakPPOFDMhPixv7ABNgjeTg4jKyfU?usp=sharing)   |
|   SMTR    |     U14M      | 51.0 | [link](https://drive.google.com/drive/folders/1-K5O0d0q9fhY5fJvU6nn5fFFtSMnbE_-?usp=drive_link) |
| FocalSVTR |     U14M      | 42.1 |  [link](https://drive.google.com/drive/folders/100xF5wFr7xSCVBYM1h_0d_8xv5Qeqobp?usp=sharing)   |

- Test on Common Benchmarks from [PARSeq](https://github.com/baudm/parseq):

|   Model   | Training Data | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |    Config&Model&Log     |
| :-------: | :-----------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :---------------------: |
|   SMTR    |      Syn      |     97.4     | 94.9 |      97.4       |     88.4      | 89.9 |  96.2  | 94.02 | Same as the above table |
|   SMTR    |     U14M      |     98.3     | 97.4 |      99.0       |     90.1      | 92.7 |  97.9  | 95.90 | Same as the above table |
| FocalSVTR |     U14M      |     97.3     | 96.3 |      98.2       |     87.4      | 88.4 |  96.2  | 93.97 | Same as the above table |

- Test on Union14M-L benchmark from [Union14M](https://github.com/Mountchicken/Union14M/).

|   Model   | Traing Data | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :-------: | :---------: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
|   SMTR    |     Syn     | 74.2  |        30.6         |   58.5   |    67.6     |  79.6   |      75.1       |  67.9   | 64.79 | Same as the above table |
|   SMTR    |    U14M     | 89.1  |        87.7         |   76.8   |    83.9     |  84.6   |      89.3       |  83.7   | 85.00 | Same as the above table |
| FocalSVTR |    U14M     | 77.7  |        62.4         |   65.7   |    78.6     |  71.6   |      81.3       |  79.2   | 73.80 | Same as the above table |

- Training and test on Chinese dataset, from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|     Model     | Scene | Web  | Document | Handwriting |  Avg  |                                        Config&Model&Log                                         |
| :-----------: | :---: | :--: | :------: | :---------: | :---: | :---------------------------------------------------------------------------------------------: |
| SMTR  w/o Aug | 79.8  | 80.6 |   99.1   |    61.9     | 80.33 | [link](https://drive.google.com/drive/folders/1v8CK5GIu7wunnD5jFh2bLbusjyHeban5?usp=drive_link) |
|  SMTR w/ Aug  | 83.4  | 83.0 |   99.3   |    65.1     | 82.68 | [link](https://drive.google.com/drive/folders/1SQnwSm0bOBQ0eMKKD08F_4Blkjie_3la?usp=drive_link) |

Download all Configs, Models, and Logs from [Google Drive](https://drive.google.com/drive/folders/1dCuaWwCLP9xIHgy-7NtpeDLOvgk9NoKE?usp=drive_link).

<a name="2"></a>

## 2. Environment

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
git clone -b develop https://github.com/Topdu/OpenOCR.git
cd OpenOCR
# A100 Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.8
conda activate openocr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Dataset Preparation

- [English dataset download](https://github.com/baudm/parseq)

- [Union14M-L download](https://github.com/Mountchicken/Union14M)

- [Chinese dataset download](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

- [LTB download](https://drive.google.com/drive/folders/1NChdlw7ustbXtlFBmh_0xnHvRkffb9Ge?usp=sharing)

The expected filesystem structure is as follows:

```
benchmark_bctr
├── benchmark_bctr_test
│   ├── document_test
│   ├── handwriting_test
│   ├── scene_test
│   └── web_test
└── benchmark_bctr_train
    ├── document_train
    ├── handwriting_train
    ├── scene_train
    └── web_train
evaluation
├── CUTE80
├── IC13_857
├── IC15_1811
├── IIIT5k
├── SVT
└── SVTP
OpenOCR
synth
├── MJ
│   ├── test
│   ├── train
│   └── val
└── ST
test # from PARSeq
├── ArT
├── COCOv1.4
├── CUTE80
├── IC13_1015
├── IC13_1095
├── IC13_857
├── IC15_1811
├── IC15_2077
├── IIIT5k
├── SVT
├── SVTP
└── Uber
u14m # lmdb format
├── artistic
├── contextless
├── curve
├── general
├── multi_oriented
├── multi_words
└── salient
ltb # download link: https://drive.google.com/drive/folders/1NChdlw7ustbXtlFBmh_0xnHvRkffb9Ge?usp=sharing
Union14M-LMDB-L # lmdb format
├── train_challenging
├── train_easy
├── train_hard
├── train_medium
└── train_normal
```

<a name="3"></a>

## 3. Model Training / Evaluation

Training:

```shell
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/smtr/focalsvtr_smtr.yml
# For RTX 4090
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/smtr/focalsvtr_smtr.yml
```

Evaluation:

```shell
# en
python tools/eval_rec_all_ratio.py --c configs/rec/smtr/focalsvtr_smtr.yml
# long text
python tools/eval_rec_all_long_simple.py --c configs/rec/smtr/focalsvtr_smtr_long.yml
# ch
python tools/eval_rec_all_ch.py --c configs/rec/smtr/focalsvtr_smtr_ch.yml
```

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@article{Du2024SMTR,
  title     = {Out of Length Text Recognition with Sub-String Matching},
  author    = {Yongkun Du, Zhineng Chen, Caiyan Jia, Xieping Gao, Yu-Gang Jiang},
  journal   = {CoRR},
  eprinttype = {arXiv},
  primaryClass={cs.CV},
  volume    = {abs/2407.12317},
  year      = {2024},
  url       = {https://arxiv.org/abs/2407.12317}
}
```

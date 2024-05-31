# IGTR

- [1. Introduction](#1)
- [2. Environment](#2)
- [3. Model Training / Evaluation](#3)
    - [3.1 Training](#3-1)
    - [3.2 Evaluation](#3-2)
    - [3.3 Prediction](#3-3)

<a name="1"></a>
## 1. Introduction

Paper:
> [Instruction-Guided Scene Text Recognition](https://arxiv.org/abs/2401.17851)
> Yongkun Du, Zhineng Chen, Yuchen Su, Caiyan Jia, Yu-Gang Jiang


<a name="model"></a>
Visual-and-language models have shown appealing performance in visual tasks recently, as free-form text-guided training evokes the ability to understand fine-grained visual content. However, current models cannot be trivially applied to scene text recognition (STR) due to the composition difference between natural and text images. We propose a novel instruction-guided scene text recognition (IGTR) paradigm that, for the first time, formulates STR as an instruction learning problem and achieves text recognition by understanding character attributes, e.g., character frequency, position, etc. IGTR first devises instruction triplets of $\left \langle condition,question,answer\right \rangle$, providing rich and diverse descriptions of character attributes. Then, we develop an architecture with a dedicated cross-modal learning module and multi-task answer head. It effectively learns character attributes and guides nuanced text image understanding by answering questions, which considerably differs from current methods that maximize the character prediction probability. Based on the learned character attributes, IGTR can implement different recognition pipelines by merely using different recognition instructions. Moreover, incorporating the advantage of easy tuning of the instruction learning paradigm, IGTR offers an elegant way which simply adjusts the instruction sampling rule to tackle rarely appearing and morphologically similar character recognition, which were previous challenges. Experiments on English and Chinese benchmarks show that IGTR outperforms existing models by significant margins in all aforementioned scenarios.

<a name="model"></a>
The accuracy (%) and model files of IGTR on the public dataset of scene text recognition are as follows:

* Trained on Synth dataset(MJ+ST), test on Common Benchmarks, training and test datasets both from [PARSeq](https://github.com/baudm/parseq).


|    Model      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      Config&Model&Log       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| IGTR-PD  | 97.6 | 95.2 | 97.6 | 88.4 | 91.6 | 95.5 | 94.30 | [link](https://drive.google.com/drive/folders/1Pv0CW2hiWC_dIyaB74W1fsXqiX3z5yXA?usp=drive_link) |
| IGTR-AR | 98.6 | 95.7 | 98.2 | 88.4 | 92.4 | 95.5 | 94.78 | as above |


* Test on Union14M-L benchmark, from [Union14M](https://github.com/Mountchicken/Union14M/).

|    Model      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     Config&Model&Log       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| IGTR-PD   |  76.9  |  30.6   |  59.1  |  63.3   |  77.8    |    62.5   |  66.7   | 62.40 | Same as the above table |
| IGTR-AR |  78.4  |  31.9  | 61.3    |  66.5  | 80.2  |   69.3 | 67.9  |  65.07 | as above |


* Trained on Union14M-L training dataset.

|    Model      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      Config&Model&Log       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| IGTR-PD  |  97.7    |   97.7   |  98.3    |  89.8    |  93.7    |  97.9    | 95.86  | [link](https://drive.google.com/drive/folders/1ZGlzDqEzjrBg8qG2wBkbOm3bLRzFbTzo?usp=drive_link) |
| IGTR-AR |  98.1    |  98.4    |   98.7    |   90.5     | 94.9   |    98.3  | 96.48   | as above |
| IGTR-PD-60ep | 97.9 | 98.3 | 99.2 | 90.8 | 93.7 | 97.6 | 96.24 | [link](https://drive.google.com/drive/folders/1ik4hxZDRsjU1RbCA19nwE45Kg1bCnMoa?usp=drive_link) |
| IGTR-AR-60ep | 98.4 | 98.1 | 99.3 | 91.5 | 94.3 | 97.6 | 96.54 | as above |
| IGTR-PD-PT | 98.6 | 98.0 | 99.1 | 91.7 | 96.8 | 99.0 | 97.20 | [link](https://drive.google.com/drive/folders/1QM0EWV66IfYI1G0Xm066V2zJA62hH6-1?usp=drive_link) |
| IGTR-AR-PT | 98.8 | 98.3 | 99.2 | 92.0 | 96.8 | 99.0 | 97.34 | as above |

|    Model      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     Config&Model&Log       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| IGTR-PD  | 88.1 | 89.9 | 74.2 | 80.3 | 82.8 | 79.2  | 83.0 | 82.51 | Same as the above table |
| IGTR-AR | 90.4 | 91.2 | 77.0 | 82.4 | 84.7 | 84.0 | 84.4  | 84.86| as above |
| IGTR-PD-60ep | 90.0 | 92.1 | 77.5 | 82.8 | 86.0 | 83.0 | 84.8 | 85.18 | Same as the above table |
| IGTR-AR-60ep | 91.0 | 93.0 | 78.7 | 84.6 | 87.3 | 84.8 | 85.6 | 86.43 | as above |
| IGTR-PD-PT | 92.4 | 92.1 | 80.7 | 83.6 | 87.7 | 86.9 | 85.0 | 86.92 | Same as the above table |
| IGTR-AR-PT | 93.0 | 92.9 | 81.3 | 83.4 | 88.6 | 88.7 | 85.6 | 87.65 | as above |

* Trained and test on Chinese dataset, from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|    Model      | Scene | Web | Document | Handwriting | Avg |      Config&Model&Log       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|
|IGTR-PD      | 73.1 | 74.8 | 98.6 |  52.5 | 74.75  |  |
|IGTR-AR    | 75.1 | 76.4 | 98.7 |  55.3 | 76.37  |  |
|IGTR-PD-TS     | 73.5 | 75.9 | 98.7 |  54.5 | 75.65  | [link](https://drive.google.com/drive/folders/1H3VRdGHjhawd6fkSC-qlBzVzvYYTpHRg?usp=drive_link)  |
|IGTR-AR-TS     | 75.6 | 77.0 | 98.8 |  57.3 | 77.17  | as above |
|IGTR-PD-Aug     | 79.5 | 80.0 | 99.4 |  58.9 | 79.45  | [link](https://drive.google.com/drive/folders/1XFQkCILwcFwA7iYyQY9crnrouaI5sqcZ?usp=drive_link)  |
|IGTR-AR-Aug     | 82.0 | 81.7 | 99.5 | 63.8 | 81.74  | as above |

Download all Configs, Models, and Logs from [Google Drive](https://drive.google.com/drive/folders/1mSRDg9Mj5R6PspAdFGXZHDHTCQmjkd8d?usp=drive_link).

<a name="2"></a>
## 2. Environment

* [PyTorch](http://pytorch.org/) version >= 1.13.0
* Python version >= 3.7

```shell
git clone -b develop https://github.com/Topdu/OpenOCR.git
cd OpenOCR
# A100 Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.9
conda activate CCD
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Dataset Preparation

[English dataset download](https://github.com/baudm/parseq)

[Union14M-L download](https://github.com/Mountchicken/Union14M)

[Chinese dataset download](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

The expected filesystem structure is as follows:
```
OpenOCR
evaluation
├── CUTE80
├── IC13_857
├── IC15_1811
├── IIIT5k
├── SVT
└── SVTP
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
# The configuration file is available from the link provided in the table above.
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/train_rec.py --c PATH/svtr_base_igtr_XXX.yml
# For RTX 4090
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/train_rec.py --c PATH/svtr_base_igtr_XXX.yml
```

Evaluation:


```shell
# The configuration file is available from the link provided in the table above.
# en
python tools/eval_rec_all_ratio.py --c PATH/svtr_base_igtr_syn.yml
# ch
python tools/eval_rec_all_ch.py --c PATH/svtr_base_igtr_ch_aug.yml
```


## 引用

```bibtex
@article{Du2024IGTR,
  title     = {Instruction-Guided Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Su, Yuchen and Jia, Caiyan and Jiang, Yu-Gang},
  journal   = {CoRR},
  eprinttype = {arXiv},
  primaryClass={cs.CV},
  volume    = {abs/2401.17851},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.17851}
}
```
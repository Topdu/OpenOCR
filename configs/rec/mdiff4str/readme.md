# MDiff4STR

- [MDiff4STR](#mdiff4str)
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

> [MDiff4STR: Mask Diffusion Model for Scene Text Recognition](https://arxiv.org/abs/2512.01422)
> Yongkun Du, Miaomiao Zhao, Songlin Fan, Zhineng Chen\*, Caiyan Jia, Yu-Gang Jiang

<a name="model"></a>
Mask Diffusion Models (MDMs) have recently emerged as a promising alternative to auto-regressive models (ARMs) for vision-language tasks, owing to their flexible balance of efficiency and accuracy. In this paper, for the first time, we introduce MDMs into the Scene Text Recognition (STR) task. We show that vanilla MDM lags behind ARMs in terms of accuracy, although it improves recognition efficiency. To bridge this gap, we propose MDiff4STR, a Mask Diffusion model enhanced with two key improvement strategies tailored for STR. Specifically, we identify two key challenges in applying MDMs to STR: noising gap between training and inference, and overconfident predictions during inference. Both significantly hinder the performance of MDMs. To mitigate the first issue, we develop six noising strategies that better align training with inference behavior. For the second, we propose a token-replacement noise mechanism that provides a non-mask noise type, encouraging the model to reconsider and revise overly confident but incorrect predictions. We conduct extensive evaluations of MDiff4STR on both standard and challenging STR benchmarks, covering diverse scenarios including irregular, artistic, occluded, and Chinese text, as well as whether the use of pretraining. Across these settings, MDiff4STR consistently outperforms popular STR models, surpassing state-of-the-art ARMs in accuracy, while maintaining fast inference with only three denoising steps.

### 1.1 Models and Results

The accuracy (%) and model files of MDiff4STR on the public dataset of scene text recognition are as follows:

Download all Configs, Models, and Logs from [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR).

- Test on Common Benchmarks from [PARSeq](https://github.com/baudm/parseq):

|    Model    |                        Training Data                         | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                            Config&Model&Log                            |
| :---------: | :----------------------------------------------------------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :--------------------------------------------------------------------: |
| MDiff4STR-B |                  Synthetic datasets (MJ+ST)                  |     98.1     | 95.8 |      98.2       |     88.1      | 91.6 |  96.9  | 94.81 | [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_base_syn) |
| MDiff4STR-S | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) |     99.0     | 98.3 |      98.4       |     90.2      | 94.9 |  97.6  | 96.38 | [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_small) |
| MDiff4STR-B | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) |     99.2     | 98.3 |      99.1       |     91.6      | 97.1 |  98.6  | 97.30 | [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_base) |

- Test on Union14M-L benchmark from [Union14M](https://github.com/Mountchicken/Union14M/).

|    Model    |                         Traing Data                          | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :---------: | :----------------------------------------------------------: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
| MDiff4STR-B |                  Synthetic datasets (MJ+ST)                  | 79.8  |        30.2         |   64.0   |    68.2     |  80.8   |      66.9       |  67.4   | 65.33 | Same as the above table |
| MDiff4STR-S | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 91.8  |        94.0         |   80.2   |    85.1     |  87.3   |      84.8       |  85.9   | 87.03 | Same as the above table |
| MDiff4STR-B | [Union14M-L-Filter](../../../docs/svtrv2.md#dataset-details) | 93.7  |        94.4         |   82.1   |    86.1     |  87.7   |      88.3       |  86.8   | 88.44 | Same as the above table |

- Training and test on Chinese dataset, from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|    Model    | Scene | Web  | Document | Handwriting |  Avg  |                                     Config&Model&Log                                     |
| :---------: | :---: | :--: | :------: | :---------: | :---: | :--------------------------------------------------------------------------------------: |
| MDiff4STR-S | 85.2  | 84.1 |   99.6   |    66.7     | 83.89 | [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_small_ch) |
| MDiff4STR-B | 85.7  | 84.7 |   99.6   |    67.0     | 84.25 | [HuggingFace Model](https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_base_ch)  |

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
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml

# For Multi RTX 4090
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=23333 --nproc_per_node=4 tools/train_rec.py --c configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml
# 20epoch runs for about 6 hours
```

### Evaluation

```shell
# Common, Union14M-Benchmark, OST
python tools/eval_rec_all_en.py --c configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml
```

After a successful run, the results are saved in a csv file in `output_dir` in the config file.

### Inference

```shell
python tools/infer_rec.py --c configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

### Latency Measurement

Firstly, downloading the IIIT5K images from [Google Drive](https://drive.google.com/drive/folders/1Po1LSBQb87DxGJuAgLNxhsJ-pdXxpIfS?usp=drive_link). Then, running the following command:

```shell
python tools/infer_rec.py --c configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml --o Global.infer_img=../iiit5k_test_image
```

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@inproceedings{Du2025MDiff5STR,
      title={MDiff4STR: Mask Diffusion Model for Scene Text Recognition},
      author={Yongkun Du and Miaomiao Zhao and Songlin Fan and Zhineng Chen and Caiyan Jia and Yu-Gang Jiang},
      booktitle={AAAI Oral},
      year={2025},
}
```

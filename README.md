<div align="center">

<h1> OpenOCR: A general OCR system with accuracy and efficiency </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟. </h5>

<a href="https://github.com/Topdu/OpenOCR/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/Topdu/OpenOCR"></a>
<a href='https://arxiv.org/abs/2411.15858'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://huggingface.co/spaces/topdu/OpenOCR-Demo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://modelscope.cn/studios/topdktu/OpenOCR-Demo" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/Topdu/OpenOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/Topdu/OpenOCR?color=9ea"></a>
<a href="https://pepy.tech/project/openocr"><img src="https://static.pepy.tech/personalized-badge/openocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Clone%20downloads"></a>
<a href="https://github.com/Topdu/OpenOCR/stargazers"><img src="https://img.shields.io/github/stars/Topdu/OpenOCR?color=ccf"></a>
<a href="https://pypi.org/project/openocr-python/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openocr-python"><img src="https://img.shields.io/pypi/dm/openocr-python?label=PyPI%20downloads"></a>

<a href="#quick-start"> 🚀 Quick Start </a> | English | [简体中文](./README_ch.md)

</div>

______________________________________________________________________

We aim to establish a unified benchmark for training and evaluating models in scene text detection and recognition. Building on this benchmark, we introduce a general OCR system with accuracy and efficiency, **OpenOCR**. This repository also serves as the official codebase of the OCR team from the [FVL Laboratory](https://fvl.fudan.edu.cn), Fudan University.

We sincerely welcome the researcher to recommend OCR or relevant algorithms and point out any potential factual errors or bugs. Upon receiving the suggestions, we will promptly evaluate and critically reproduce them. We look forward to collaborating with you to advance the development of OpenOCR and continuously contribute to the OCR community!

## Features

- 🔥**OpenOCR: A general OCR system with accuracy and efficiency**
  - ⚡\[[Quick Start](#quick-start)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[Local Demo](#local-demo)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
  - [Introduction](./docs/openocr.md)
    - A practical OCR system building on SVTRv2.
    - Outperforms [PP-OCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html) baseline by 4.5% on the [OCR competition leaderboard](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard) in terms of accuracy, while preserving quite similar inference speed.
    - [x] Supports Chinese and English text detection and recognition.
    - [x] Provides server model and mobile model.
    - [x] Fine-tunes OpenOCR on a custom dataset: [Fine-tuning Det](./docs/finetune_det.md), [Fine-tuning Rec](./docs/finetune_rec.md).
    - [x] [ONNX model export for wider compatibility](#export-onnx-model).
- 🔥**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
  - \[[Paper](https://arxiv.org/abs/2411.15858)\] \[[Doc](./configs/rec/svtrv2/)\] \[[Model](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Datasets](./docs/svtrv2.md#downloading-datasets)\] \[[Config, Training and Inference](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[Benchmark](./docs/svtrv2.md#results-benchmark--configs--checkpoints)\]
  - [Introduction](./docs/svtrv2.md)
    - A unified training and evaluation benchmark (on top of [Union14M](https://github.com/Mountchicken/Union14M?tab=readme-ov-file#3-union14m-dataset)) for Scene Text Recognition
    - Supports 24 Scene Text Recognition methods trained from scratch on the large-scale real dataset [Union14M-L-Filter](./docs/svtrv2.md#dataset-details), and will continue to add the latest methods.
    - Improves accuracy by 20-30% compared to models trained based on synthetic datasets.
    - Towards Arbitrary-Shaped Text Recognition and Language modeling with a Single Visual Model.
    - Surpasses Attention-based Encoder-Decoder Methods across challenging scenarios in terms of accuracy and speed
  - [Get Started](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch) with training a SOTA Scene Text Recognition model from scratch.

## Ours STR algorithms

- [**SVTRv2**](./configs/rec/svtrv2) (*Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition,* ICCV 2025. [Doc](./configs/rec/svtrv2/), [Paper](https://arxiv.org/abs/2411.15858))
- [**IGTR**](./configs/rec/igtr/) (*Yongkun Du, Zhineng Chen\*, Yuchen Su, Caiyan Jia, Yu-Gang Jiang. Instruction-Guided Scene Text Recognition,* TPAMI 2025. [Doc](./configs/rec/igtr), [Paper](https://ieeexplore.ieee.org/document/10820836))
- [**CPPD**](./configs/rec/cppd/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Chenxia Li, Yuning Du, Yu-Gang Jiang. Context Perception Parallel Decoder for Scene Text Recognition,* TPAMI 2025. [PaddleOCR Doc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_cppd.en.md), [Paper](https://ieeexplore.ieee.org/document/10902187))
- [**SMTR&FocalSVTR**](./configs/rec/smtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang. Out of Length Text Recognition with Sub-String Matching,* AAAI 2025. [Doc](./configs/rec/smtr/), [Paper](https://arxiv.org/abs/2407.12317))
- [**DPTR**](./configs/rec/dptr/) (*Shuai Zhao, Yongkun Du, Zhineng Chen\*, Yu-Gang Jiang. Decoder Pre-Training with only Text for Scene Text Recognition,* ACM MM 2024. [Paper](https://arxiv.org/abs/2408.05706))
- [**CDistNet**](./configs/rec/cdistnet/) (*Tianlun Zheng, Zhineng Chen\*, Shancheng Fang, Hongtao Xie, Yu-Gang Jiang. CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition,* IJCV 2024. [Paper](https://link.springer.com/article/10.1007/s11263-023-01880-0))
- **MRN** (*Tianlun Zheng, Zhineng Chen\*, Bingchen Huang, Wei Zhang, Yu-Gang Jiang. MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition,* ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_MRN_Multiplexed_Routing_Network_for_Incremental_Multilingual_Text_Recognition_ICCV_2023_paper.html), [Code](https://github.com/simplify23/MRN))
- **TPS++** (*Tianlun Zheng, Zhineng Chen\*, Jinfeng Bai, Hongtao Xie, Yu-Gang Jiang. TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition,* IJCAI 2023. [Paper](https://arxiv.org/abs/2305.05322), [Code](https://github.com/simplify23/TPS_PP))
- [**SVTR**](./configs/rec/svtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model,* IJCAI 2022 (Long). [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_svtr.md), [Paper](https://www.ijcai.org/proceedings/2022/124))
- [**NRTR**](./configs/rec/nrtr/) (*Fenfen Sheng, Zhineng Chen, Bo Xu. NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition,* ICDAR 2019. [Paper](https://arxiv.org/abs/1806.00926))

## Recent Updates

- **2025.07.10**: Our paper [SVTRv2](https://arxiv.org/abs/2411.15858) is accepted by ICCV 2025. Accessible in [Doc](./configs/rec/svtrv2/).

- **2025.03.24**: 🔥 Releasing the feature of fine-tuning OpenOCR on a custom dataset: [Fine-tuning Det](./docs/finetune_det.md), [Fine-tuning Rec](./docs/finetune_rec.md)

- **2025.03.23**: 🔥 Releasing the feature of [ONNX model export for wider compatibility](#export-onnx-model).

- **2025.02.22**: Our paper [CPPD](https://ieeexplore.ieee.org/document/10902187) is accepted by TPAMI. Accessible in [Doc](./configs/rec/cppd/) and [PaddleOCR Doc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_cppd.en.md).

- **2024.12.31**: Our paper [IGTR](https://ieeexplore.ieee.org/document/10820836) is accepted by TPAMI. Accessible in [Doc](./configs/rec/igtr/).

- **2024.12.16**: Our paper [SMTR](https://arxiv.org/abs/2407.12317) is accepted by AAAI 2025. Accessible in [Doc](./configs/rec/smtr/).

- **2024.12.03**: The pre-training code for [DPTR](https://arxiv.org/abs/2408.05706) is merged.

- **🔥 2024.11.23 release notes**:

  - **OpenOCR: A general OCR system with accuracy and efficiency**
    - ⚡\[[Quick Start](#quick-start)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[Local Demo](#local-demo)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
    - [Introduction](./docs/openocr.md)
  - **SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
    - \[[Paper](https://arxiv.org/abs/2411.15858)\] \[[Doc](./configs/rec/svtrv2/)\] \[[Model](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Datasets](./docs/svtrv2.md#downloading-datasets)\] \[[Config, Training and Inference](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[Benchmark](./docs/svtrv2.md#results--configs--checkpoints)\]
    - [Introduction](./docs/svtrv2.md)
    - [Get Started](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch) with training a SOTA Scene Text Recognition model from scratch.

## Quick Start

**Note**: OpenOCR supports inference using both the ONNX and Torch frameworks, with the dependency environments for the two frameworks being isolated. When using ONNX for inference, there is no need to install Torch, and vice versa.

### 1. ONNX Inference

#### Install OpenOCR and Dependencies:

```shell
pip install openocr-python
pip install onnxruntime
```

#### Usage:

```python
from openocr import OpenOCR
onnx_engine = OpenOCR(backend='onnx', device='cpu')
img_path = '/path/img_path or /path/img_file'
result, elapse = onnx_engine(img_path)
```

### 2. Pytorch inference

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

After installing dependencies, the following two installation methods are available. Either one can be chosen.

#### 2.1. Python Modules

**Install OpenOCR**:

```shell
pip install openocr-python
```

**Usage**:

```python
from openocr import OpenOCR
engine = OpenOCR()
img_path = '/path/img_path or /path/img_file'
result, elapse = engine(img_path)

# Server mode
# engine = OpenOCR(mode='server')
```

#### 2.2. Clone this repository:

```shell
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

##### Export ONNX model

```shell
pip install onnx
python tools/toonnx.py --c configs/rec/svtrv2/repsvtr_ch.yml --o Global.device=cpu
python tools/toonnx.py --c configs/det/dbnet/repvit_db.yml --o Global.device=cpu
```

##### Inference with ONNXRuntime

```shell
pip install onnxruntime
# OpenOCR system: Det + Rec model
python tools/infer_e2e.py --img_path=/path/img_fold or /path/img_file --backend=onnx --device=cpu
# Det model
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_fold or /path/img_file
# Rec model
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_fold or /path/img_file
```

#### Local Demo

```shell
pip install gradio==4.20.0
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/OCR_e2e_img.tar
tar xf OCR_e2e_img.tar
# start demo
python demo_gradio.py
```

## Reproduction schedule:

### Scene Text Recognition

| Method                                        | Venue                                                                                          | Training | Evaluation | Contributor                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------- | ---------- | ------------------------------------------- |
| [CRNN](./configs/rec/svtrs/)                  | [TPAMI 2016](https://arxiv.org/abs/1507.05717)                                                 | ✅       | ✅         |                                             |
| [ASTER](./configs/rec/aster/)                 | [TPAMI 2019](https://ieeexplore.ieee.org/document/8395027)                                     | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [NRTR](./configs/rec/nrtr/)                   | [ICDAR 2019](https://arxiv.org/abs/1806.00926)                                                 | ✅       | ✅         |                                             |
| [SAR](./configs/rec/sar/)                     | [AAAI 2019](https://aaai.org/papers/08610-show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition/) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [MORAN](./configs/rec/moran/)                 | [PR 2019](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300263)             | ✅       | ✅         |                                             |
| [DAN](./configs/rec/dan/)                     | [AAAI 2020](https://arxiv.org/pdf/1912.10205)                                                  | ✅       | ✅         |                                             |
| [RobustScanner](./configs/rec/robustscanner/) | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3160_ECCV_2020_paper.php)   | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [AutoSTR](./configs/rec/autostr/)             | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690732.pdf)            | ✅       | ✅         |                                             |
| [SRN](./configs/rec/srn/)                     | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [SEED](./configs/rec/seed/)                   | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html) | ✅       | ✅         |                                             |
| [ABINet](./configs/rec/abinet/)               | [CVPR 2021](https://openaccess.thecvf.com//content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [VisionLAN](./configs/rec/visionlan/)         | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| PIMNet                                        | [ACM MM 2021](https://dl.acm.org/doi/10.1145/3474085.3475238)                                  |          |            | TODO                                        |
| [SVTR](./configs/rec/svtrs/)                  | [IJCAI 2022](https://www.ijcai.org/proceedings/2022/124)                                       | ✅       | ✅         |                                             |
| [PARSeq](./configs/rec/parseq/)               | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)            | ✅       | ✅         |                                             |
| [MATRN](./configs/rec/matrn/)                 | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)            | ✅       | ✅         |                                             |
| [MGP-STR](./configs/rec/mgpstr/)              | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)            | ✅       | ✅         |                                             |
| [LPV](./configs/rec/lpv/)                     | [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0189.pdf)                                  | ✅       | ✅         |                                             |
| [MAERec](./configs/rec/maerec/)(Union14M)     | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [LISTER](./configs/rec/lister/)               | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [CDistNet](./configs/rec/cdistnet/)           | [IJCV 2024](https://link.springer.com/article/10.1007/s11263-023-01880-0)                      | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [BUSNet](./configs/rec/busnet/)               | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28402)                            | ✅       | ✅         |                                             |
| DCTC                                          | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28575)                            |          |            | TODO                                        |
| [CAM](./configs/rec/cam/)                     | [PR 2024](https://arxiv.org/abs/2402.13643)                                                    | ✅       | ✅         |                                             |
| [OTE](./configs/rec/ote/)                     | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.html) | ✅       | ✅         |                                             |
| CFF                                           | [IJCAI 2024](https://arxiv.org/abs/2407.05562)                                                 |          |            | TODO                                        |
| [DPTR](./configs/rec/dptr/)                   | [ACM MM 2024](https://arxiv.org/abs/2408.05706)                                                |          |            | [fd-zs](https://github.com/fd-zs)           |
| VIPTR                                         | [ACM CIKM 2024](https://arxiv.org/abs/2401.10110)                                              |          |            | TODO                                        |
| [IGTR](./configs/rec/igtr/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10820836)                                    | ✅       | ✅         |                                             |
| [SMTR](./configs/rec/smtr/)                   | [AAAI 2025](https://arxiv.org/abs/2407.12317)                                                  | ✅       | ✅         |                                             |
| [CPPD](./configs/rec/cppd/)                   | [TPAMI Online Access](https://ieeexplore.ieee.org/document/10902187)                           | ✅       | ✅         |                                             |
| [FocalSVTR-CTC](./configs/rec/svtrs/)         | [2024](https://arxiv.org/abs/2407.12317)                                                       | ✅       | ✅         |                                             |
| [SVTRv2](./configs/rec/svtrv2/)               | [2024](https://arxiv.org/abs/2411.15858)                                                       | ✅       | ✅         |                                             |
| [ResNet+Trans-CTC](./configs/rec/svtrs/)      |                                                                                                | ✅       | ✅         |                                             |
| [ViT-CTC](./configs/rec/svtrs/)               |                                                                                                | ✅       | ✅         |                                             |

#### Contributors

______________________________________________________________________

Yiming Lei ([pretto0](https://github.com/pretto0)), Xingsong Ye ([YesianRohn](https://github.com/YesianRohn)), and Shuai Zhao ([fd-zs](https://github.com/fd-zs)) from the [FVL Laboratory](https://fvl.fudan.edu.cn), Fudan University, with guidance from Dr. Zhineng Chen ([Homepage](https://zhinchenfd.github.io/)), completed the majority work of the algorithm reproduction. Grateful for their outstanding contributions.

### Scene Text Detection (STD)

TODO

### Text Spotting

TODO

______________________________________________________________________

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

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!

# OpenOCR

We aim to establishing a unified benchmark for training and evaluating models for scene text detection and recognition. Based on this benchmark, we introduce an accurate and efficient general OCR system, OpenOCR. Additionally, this repository will serve as the official codebase for the OCR team from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University.

We sincerely welcome the researcher to recommend OCR or relevant algorithms and point out any potential factual errors or bugs. Upon receiving the suggestions, we will promptly evaluate and critically reproduce them. We look forward to collaborating with you to advance the development of OpenOCR and continuously contribute to the OCR community!

## Features

- 🔥**OpenOCR: A general OCR system for accuracy and efficiency**
  - ⚡\[[Quick Start](<>)\] \[[Demo](<>)\]
  - [Introduction](./docs/openocr.md)
    - A practical version of the model builds on SVTRv2.
    - Outperforming [PP-OCRv4](<>) released by [PaddleOCR](<>) by 4.5% on the [OCR competition leaderboard](<>).
    - [x] Supporting Chinese and English text detection and recognition.
    - [x] Providing server model and mobile model.
    - [ ] Fine-tuning OpenOCR on a custom dataset
    - [ ] Export to ONNX engine
- 🔥**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
  - \[[Paper](./configs/rec/svtrv2/SVTRv2.pdf)\] \[[Model](../configs/rec/svtrv2/)\] \[[Config, Training and Inference](../configs/rec/svtrv2/)\]
  - [Introduction](./docs/svtrv2.md)
    - Developing a unified training and evaluation benchmark for Scene Text Recognition
    - Supporting for 24 Scene Text Recognition methods trained from scratch on large-scale real datasets, and will continue to add the latest methods.
    - Improving results by 20-30% compared to training on synthetic datasets.
    - Towards Arbitrary-Shaped Text Recognition and Language modeling with a Single Visual Model.
    - Surpasses Attention-based Decoder Methods across challenging scenarios in terms of accuracy and speed
  - [Get Started](./docs/svtrv2.md) with training a SoTA Scene Text Recognition model from scratch.

## Ours STR algorithms

- [**DPTR**](<>) (*Shuai Zhao, Yongkun Du, Zhineng Chen\*, Yu-Gang Jiang. Decoder Pre-Training with only Text for Scene Text Recognition,* ACM MM 2024. [paper](https://arxiv.org/abs/2408.05706))
- [**IGTR**](./configs/rec/igtr/) (*Yongkun Du, Zhineng Chen\*, Yuchen Su, Caiyan Jia, Yu-Gang Jiang. Instruction-Guided Scene Text Recognition,* Under TPAMI minor revison 2024. [Doc](./configs/rec/igtr/readme.md), [paper](https://arxiv.org/abs/2401.17851))
- [**SVTRv2**](./configs/rec/svtrv2) (*Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition,* 2024. [paper](./configs/rec/svtrv2/SVTRv2.pdf))
- [**SMTR&FocalSVTR**](./configs/rec/smtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang. Out of Length Text Recognition with Sub-String Matching,* 2024. [paper](https://arxiv.org/abs/2407.12317))
- [**CDistNet**](./configs/rec/cdistnet/) (*Tianlun Zheng, Zhineng Chen\*, Shancheng Fang, Hongtao Xie, Yu-Gang Jiang. CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition,* IJCV 2024. [paper](https://link.springer.com/article/10.1007/s11263-023-01880-0))
- **MRN** (*Tianlun Zheng, Zhineng Chen\*, Bingchen Huang, Wei Zhang, Yu-Gang Jiang. MRN: Multiplexed routing network for incremental multilingual text recognition,* ICCV 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_MRN_Multiplexed_Routing_Network_for_Incremental_Multilingual_Text_Recognition_ICCV_2023_paper.html))
- **TPS++** (*Tianlun Zheng, Zhineng Chen\*, Jinfeng Bai, Hongtao Xie, Yu-Gang Jiang. TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition,* IJCAI 2023. [paper](https://arxiv.org/abs/2305.05322))
- [**CPPD**](./configs/rec/cppd/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Chenxia Li, Yuning Du, Yu-Gang Jiang. Context Perception Parallel Decoder for Scene Text Recognition,* Under TPAMI minor revision 2023. [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_cppd.md), [paper](https://arxiv.org/abs/2307.12270))
- [**SVTR**](./configs/rec/svtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model,* IJCAI 2022 (Long). [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_svtr.md), [paper](https://www.ijcai.org/proceedings/2022/124))
- [**NRTR**](./configs/rec/nrtr/) (*Fenfen Sheng, Zhineng Chen\*, Bo Xu. NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition,* ICDAR 2019. [paper](https://arxiv.org/abs/1806.00926))

## Recent Updates

- **🔥 2024.11.23 release notes**:
  - 🔥**OpenOCR: A general OCR system for accuracy and efficiency**
    - ⚡\[[Quick Start](./docs/openocr.md)\] \[[Demo](<>)\]
    - [Introduction](./docs/openocr.md)
  - 🔥**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
    - \[[Paper](./configs/rec/svtrv2/SVTRv2.pdf)\] \[[Model](../configs/rec/svtrv2/)\] \[[Config, Training and Inference](../configs/rec/svtrv2/)\]
    - [Introduction](./docs/svtrv2.md)
    - [Get Started](./docs/svtrv2.md) with training a SoTA Scene Text Recognition model from scratch.

## ⚡[Quick Start](./docs/openocr.md)

#### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocr python==3.8
conda activate openocr
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

## Reproduction schedule:

### Scene Text Recognition

| Method                                        | Venue                                                                                          | Training | Evaluation | Contributor                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------- | ---------- | ------------------------------------------- |
| [CRNN](./configs/rec/svtrs/)                  | [TPAMI 2016](https://arxiv.org/abs/1507.05717)                                                 | ✅       | ✅         |                                             |
| [ASTER](./configs/rec/aster/)                 | [TPAMI 2019](https://ieeexplore.ieee.org/document/8395027)                                     | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [NRTR](./configs/rec/nrtr/)                   | [ICDAR 2019](https://arxiv.org/abs/1806.00926)                                                 | ✅       | ✅         |                                             |
| [SAR](./configs/rec/sar/)                     | [AAAI 2019](https://aaai.org/papers/08610-show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition/) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [MORAN](./configs/rec/moran/)                 | [PR 2019](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300263)             | ✅       | ✅         | Debug                                       |
| [DAN](./configs/rec/dan/)                     | [AAAI 2020](https://arxiv.org/pdf/1912.10205)                                                  | ✅       | ✅         |                                             |
| [RobustScanner](./configs/rec/robustscanner/) | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3160_ECCV_2020_paper.php)   | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [AutoSTR](./configs/rec/autostr/)             | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690732.pdf)            | ✅       | ✅         |                                             |
| [SRN](./configs/rec/srn/)                     | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [SEED](./configs/rec/seed/)                   | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html) | ✅       | ✅         |                                             |
| [ABINet](./configs/rec/abinet/)               | [CVPR 2021](https://openaccess.thecvf.com//content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [VisionLAN](./configs/rec/visionlan/)         | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [SVTR](./configs/rec/svtrs/)                  | [IJCAI 2022](https://www.ijcai.org/proceedings/2022/124)                                       | ✅       | ✅         |                                             |
| [PARSeq](./configs/rec/parseq/)               | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)            | ✅       | ✅         |                                             |
| [MATRN](./configs/rec/matrn/)                 | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)            | ✅       | ✅         |                                             |
| [MGP-STR](./configs/rec/mgpstr/)              | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)            | ✅       | ✅         |                                             |
| [CPPD](./configs/rec/cppd/)                   | [2023](https://arxiv.org/abs/2307.12270)                                                       | ✅       | ✅         |                                             |
| [LPV](./configs/rec/lpv/)                     | [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0189.pdf)                                  | ✅       | ✅         |                                             |
| [MAERec](./configs/rec/maerec/)(Union14M)     | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [LISTER](./configs/rec/lister/)               | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [CDistNet](./configs/rec/cdistnet/)           | [IJCV 2024](https://link.springer.com/article/10.1007/s11263-023-01880-0)                      | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [BUSNet](./configs/rec/busnet/)               | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28402)                            | ✅       | ✅         |                                             |
| DCTC                                          | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28575)                            |          |            | TODO                                        |
| [CAM](./configs/rec/cam/)                     | [PR 2024](https://arxiv.org/abs/2402.13643)                                                    | ✅       | ✅         |                                             |
| [OTE](./configs/rec/ote/)                     | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.pdf) | ✅       | ✅         |                                             |
| CFF                                           | [IJCAI 2024](https://arxiv.org/abs/2407.05562)                                                 |          |            | TODO                                        |
| DPTR                                          | [ACM MM 2024](https://arxiv.org/abs/2408.05706)                                                |          |            | TODO                                        |
| VIPTR                                         | [ACM CIKM 2024](https://arxiv.org/abs/2401.10110)                                              |          |            | TODO                                        |
| [IGTR](./configs/rec/igtr/)                   | [2024](https://arxiv.org/abs/2401.17851)                                                       | ✅       | ✅         |                                             |
| [SMTR](./configs/rec/smtr/)                   | [2024](https://arxiv.org/abs/2407.12317)                                                       | ✅       | ✅         |                                             |
| [FocalSVTR-CTC](./configs/rec/svtrs/)         | [2024](https://arxiv.org/abs/2407.12317)                                                       | ✅       | ✅         |                                             |
| [SVTRv2](./configs/rec/svtrv2/)               | [2024](./configs/rec/svtrv2/SVTRv2.pdf)                                                        | ✅       | ✅         |                                             |
| [ResNet+Trans-CTC](./configs/rec/svtrs/)      |                                                                                                | ✅       | ✅         |                                             |
| [ViT-CTC](./configs/rec/svtrs/)               |                                                                                                | ✅       | ✅         |                                             |

#### Contributors

______________________________________________________________________

Yiming Lei ([pretto0](https://github.com/pretto0)) and Xingsong Ye ([YesianRohn](https://github.com/YesianRohn)) from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University, under the guidance of Professor Zhineng Chen, completed the majority of the algorithm reproduction work. Grateful for their outstanding contributions.

### Scene Text Detection (STD)

TODO

### Text Spotting

TODO

______________________________________________________________________

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!

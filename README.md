# OpenOCR

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
    - [ ] Fine-tunes OpenOCR on a custom dataset.
    - [ ] ONNX model export for wider compatibility.
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

- [**DPTR**](./configs/rec/dptr/) (*Shuai Zhao, Yongkun Du, Zhineng Chen\*, Yu-Gang Jiang. Decoder Pre-Training with only Text for Scene Text Recognition,* ACM MM 2024. [Paper](https://arxiv.org/abs/2408.05706))
- [**IGTR**](./configs/rec/igtr/) (*Yongkun Du, Zhineng Chen\*, Yuchen Su, Caiyan Jia, Yu-Gang Jiang. Instruction-Guided Scene Text Recognition,* Under TPAMI minor revision 2024. [Doc](./configs/rec/igtr), [Paper](https://arxiv.org/abs/2401.17851))
- [**SVTRv2**](./configs/rec/svtrv2) (*Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition,* 2024. [Doc](./configs/rec/svtrv2/), [Paper](https://arxiv.org/abs/2411.15858))
- [**SMTR&FocalSVTR**](./configs/rec/smtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang. Out of Length Text Recognition with Sub-String Matching,* 2024. [Doc](./configs/rec/smtr/), [Paper](https://arxiv.org/abs/2407.12317))
- [**CDistNet**](./configs/rec/cdistnet/) (*Tianlun Zheng, Zhineng Chen\*, Shancheng Fang, Hongtao Xie, Yu-Gang Jiang. CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition,* IJCV 2024. [Paper](https://link.springer.com/article/10.1007/s11263-023-01880-0))
- **MRN** (*Tianlun Zheng, Zhineng Chen\*, Bingchen Huang, Wei Zhang, Yu-Gang Jiang. MRN: Multiplexed routing network for incremental multilingual text recognition,* ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_MRN_Multiplexed_Routing_Network_for_Incremental_Multilingual_Text_Recognition_ICCV_2023_paper.html), [Code](https://github.com/simplify23/MRN))
- **TPS++** (*Tianlun Zheng, Zhineng Chen\*, Jinfeng Bai, Hongtao Xie, Yu-Gang Jiang. TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition,* IJCAI 2023. [Paper](https://arxiv.org/abs/2305.05322), [Code](https://github.com/simplify23/TPS_PP))
- [**CPPD**](./configs/rec/cppd/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Chenxia Li, Yuning Du, Yu-Gang Jiang. Context Perception Parallel Decoder for Scene Text Recognition,* Under TPAMI minor revision 2023. [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_cppd.md), [Paper](https://arxiv.org/abs/2307.12270))
- [**SVTR**](./configs/rec/svtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model,* IJCAI 2022 (Long). [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_svtr.md), [Paper](https://www.ijcai.org/proceedings/2022/124))
- [**NRTR**](./configs/rec/nrtr/) (*Fenfen Sheng, Zhineng Chen\*, Bo Xu. NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition,* ICDAR 2019. [Paper](https://arxiv.org/abs/1806.00926))

## Recent Updates

- **2024.12.03**: The pre-training code for [DPTR](./configs/rec/dptr/) is merged.

- **🔥 2024.11.23 release notes**:

  - **OpenOCR: A general OCR system with accuracy and efficiency**
    - ⚡\[[Quick Start](#quick-start)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[Local Demo](#local-demo)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
    - [Introduction](./docs/openocr.md)
  - **SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
    - \[[Paper](https://arxiv.org/abs/2411.15858)\] \[[Doc](./configs/rec/svtrv2/)\] \[[Model](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Datasets](./docs/svtrv2.md#downloading-datasets)\] \[[Config, Training and Inference](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[Benchmark](./docs/svtrv2.md#results--configs--checkpoints)\]
    - [Introduction](./docs/svtrv2.md)
    - [Get Started](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch) with training a SOTA Scene Text Recognition model from scratch.

## Quick Start

### Dependencies:

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

### 1. Python Modules

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

### 2. Clone this repository:

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
| [CPPD](./configs/rec/cppd/)                   | [2023](https://arxiv.org/abs/2307.12270)                                                       | ✅       | ✅         |                                             |
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
| [IGTR](./configs/rec/igtr/)                   | [2024](https://arxiv.org/abs/2401.17851)                                                       | ✅       | ✅         |                                             |
| [SMTR](./configs/rec/smtr/)                   | [2024](https://arxiv.org/abs/2407.12317)                                                       | ✅       | ✅         |                                             |
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
@article{Du2024SVTRv2,
      title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
      author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
      journal={CoRR},
      volume={abs/2411.15858},
      eprinttype={arXiv},
      year={2024},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15858}
}
```

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!

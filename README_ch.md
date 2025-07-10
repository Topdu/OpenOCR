<div align="center">

<h1> OpenOCR: A general OCR system with accuracy and efficiency </h1>

<h5 align="center"> 如果您觉得本项目有帮助，请为我们点亮Star🌟 </h5>

<a href="https://github.com/Topdu/OpenOCR/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/Topdu/OpenOCR"></a>
<a href='https://arxiv.org/abs/2411.15858'><img src='https://img.shields.io/badge/论文-Arxiv-red'></a>
<a href="https://huggingface.co/spaces/topdu/OpenOCR-Demo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://modelscope.cn/studios/topdktu/OpenOCR-Demo" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/Topdu/OpenOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/Topdu/OpenOCR?color=9ea"></a>
<a href="https://pepy.tech/project/openocr"><img src="https://static.pepy.tech/personalized-badge/openocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Clone%20下载量"></a>
<a href="https://github.com/Topdu/OpenOCR/stargazers"><img src="https://img.shields.io/github/stars/Topdu/OpenOCR?color=ccf"></a>
<a href="https://pypi.org/project/openocr-python/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openocr-python"><img src="https://img.shields.io/pypi/dm/openocr-python?label=PyPI%20下载量"></a>

<a href="#快速开始"> 🚀 快速开始 </a> | 简体中文 | [English](./README.md)

</div>

______________________________________________________________________

我们致力于构建场景文本检测与识别模型的统一训练评估基准。基于此基准，我们推出了兼顾精度与效率的通用OCR系统——**OpenOCR**。本仓库同时作为复旦大学[FVL实验室](https://fvl.fudan.edu.cn)OCR团队的官方代码库。

我们诚挚欢迎研究者推荐OCR相关算法，并指出潜在的事实性错误或代码缺陷。收到建议后，我们将及时评估并严谨复现。期待与您携手推进OpenOCR发展，持续为OCR社区贡献力量！

## 核心特性

- 🔥**OpenOCR: A general OCR system with accuracy and efficiency**

  - ⚡\[[快速开始](#快速开始)\] \[[模型下载](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[本地Demo](#本地Demo)\] \[[PaddleOCR实现](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
  - [技术文档](./docs/openocr.md)
    - 基于SVTRv2构建的实用OCR系统
    - 在[OCR竞赛榜单](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard)上，精度超越[PP-OCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html)基线4.5%，推理速度保持相近
    - [x] 支持中英文文本检测与识别
    - [x] 提供服务器端(Server)与移动端(mobile)模型
    - [x] 支持自定义数据集微调: [检测模型微调](./docs/finetune_det.md), [识别模型微调](./docs/finetune_rec.md)
    - [x] [支持导出ONNX模型](#导出onnx模型)

- 🔥**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**

  - \[[论文](https://arxiv.org/abs/2411.15858)\] \[[文档](./configs/rec/svtrv2/)\] \[[模型](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[数据集](./docs/svtrv2.md#downloading-datasets)\] \[[配置/训练/推理](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[基准测试](./docs/svtrv2.md#results-benchmark--configs--checkpoints)\]
  - [技术文档](./docs/svtrv2.md)
    - 基于[Union14M](https://github.com/Mountchicken/Union14M)构建的场景文本识别统一训练评估基准
    - 支持24种场景文本识别方法在大规模真实数据集[Union14M-L-Filter](./docs/svtrv2.md#数据集详情)上的训练，将持续集成前沿方法
    - 相比基于合成数据训练的模型，精度提升20-30%
    - 单一视觉模型实现任意形状文本识别与语言建模
    - 在精度与速度上全面超越基于Attention的编解码模型
    - [从零训练SOTA模型指南](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch)

## 自研STR算法

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

## 近期更新

- **2025.07.10**: [SVTRv2](https://arxiv.org/abs/2411.15858)被ICCV 2025接收. 详见[文档](./configs/rec/svtrv2/)
- **2025.03.24**: 🔥 发布自定义数据集微调功能: [检测模型微调](./docs/finetune_det.md), [识别模型微调](./docs/finetune_rec.md)
- **2025.03.23**: 🔥 新增[ONNX模型导出功能](#导出onnx模型)
- **2025.02.22**: [CPPD](https://ieeexplore.ieee.org/document/10902187)论文被TPAMI录用，详见[文档](./configs/rec/cppd/)与[PaddleOCR文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_cppd.en.md)
- **2024.12.31**: [IGTR](https://ieeexplore.ieee.org/document/10820836)论文被TPAMI录用，详见[文档](./configs/rec/igtr/)
- **2024.12.16**: [SMTR](https://arxiv.org/abs/2407.12317)论文被AAAI 2025录用，详见[文档](./configs/rec/smtr/)
- **2024.12.03**: [DPTR](https://arxiv.org/abs/2408.05706)预训练代码合并
- **🔥 2024.11.23 重大更新**:
  - **OpenOCR通用OCR系统发布**
    - ⚡\[[快速开始](#快速开始)\] \[[模型下载](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScopeDemo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging FaceDemo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[本地Demo](#本地Demo)\] \[[PaddleOCR实现](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
    - [技术文档](./docs/openocr.md)
  - **SVTRv2论文发布**
    - \[[论文](https://arxiv.org/abs/2411.15858)\] \[[文档](./configs/rec/svtrv2/)\] \[[模型](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[数据集](./docs/svtrv2.md#downloading-datasets)\] \[[配置/训练/推理](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[基准测试](./docs/svtrv2.md#results-benchmark--configs--checkpoints)\]
    - [技术文档](./docs/svtrv2.md)
    - [从零训练SOTA模型指南](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch)

## 快速开始

**注意**: OpenOCR支持ONNX和PyTorch双框架推理，环境相互独立。使用ONNX推理时无需安装PyTorch，反之亦然。

### 1. ONNX推理

#### 安装OpenOCR及依赖:

```shell
pip install openocr-python
pip install onnxruntime
```

#### 使用示例:

```python
from openocr import OpenOCR
onnx_engine = OpenOCR(backend='onnx', device='cpu')
img_path = '/path/img_path or /path/img_file'
result, elapse = onnx_engine(img_path)
```

### 2. PyTorch推理

#### 环境依赖:

- [PyTorch](http://pytorch.org/) >= 1.13.0
- Python >= 3.7

```shell
conda create -n openocr python==3.8
conda activate openocr
# 安装GPU版本
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# 或CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### 2.1 Python包安装

**安装OpenOCR**:

```shell
pip install openocr-python
```

**使用示例**:

```python
from openocr import OpenOCR
engine = OpenOCR()
img_path = '/path/img_path or /path/img_file'
result, elapse = engine(img_path)

# Server模式
# engine = OpenOCR(mode='server')
```

#### 2.2 源码安装

```shell
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
pip install -r requirements.txt
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth
# Server识别模型
# wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth
```

**使用命令**:

```shell
# 端到端OCR系统: 检测+识别
python tools/infer_e2e.py --img_path=/path/img_path or /path/img_file
# 单独检测模型
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.infer_img=/path/img_path or /path/img_file
# 单独识别模型
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.infer_img=/path/img_path or /path/img_file
```

##### 导出ONNX模型

```shell
pip install onnx
python tools/toonnx.py --c configs/rec/svtrv2/repsvtr_ch.yml --o Global.device=cpu
python tools/toonnx.py --c configs/det/dbnet/repvit_db.yml --o Global.device=cpu
```

##### ONNXRuntime推理

```shell
pip install onnxruntime
# 端到端OCR系统
python tools/infer_e2e.py --img_path=/path/img_path or /path/img_file --backend=onnx --device=cpu
# 检测模型
python tools/infer_det.py --c ./configs/det/dbnet/repvit_db.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_path or /path/img_file
# 识别模型
python tools/infer_rec.py --c ./configs/rec/svtrv2/repsvtr_ch.yml --o Global.backend=onnx Global.device=cpu Global.infer_img=/path/img_path or /path/img_file
```

#### 本地Demo

```shell
pip install gradio==4.20.0
wget https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/OCR_e2e_img.tar
tar xf OCR_e2e_img.tar
# 启动Demo
python demo_gradio.py
```

## 算法复现计划

### 场景文本识别(STR)

| 方法                                          | 会议/期刊                                                                                        | 训练支持 | 评估支持 | 贡献者                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------- | -------- | ------------------------------------------- |
| [CRNN](./configs/rec/svtrs/)                  | [TPAMI 2016](https://arxiv.org/abs/1507.05717)                                                   | ✅       | ✅       |                                             |
| [ASTER](./configs/rec/aster/)                 | [TPAMI 2019](https://ieeexplore.ieee.org/document/8395027)                                       | ✅       | ✅       | [pretto0](https://github.com/pretto0)       |
| [NRTR](./configs/rec/nrtr/)                   | [ICDAR 2019](https://arxiv.org/abs/1806.00926)                                                   | ✅       | ✅       |                                             |
| [SAR](./configs/rec/sar/)                     | [AAAI 2019](https://aaai.org/papers/08610-show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition/) | ✅       | ✅       | [pretto0](https://github.com/pretto0)       |
| [MORAN](./configs/rec/moran/)                 | [PR 2019](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300263)               | ✅       | ✅       |                                             |
| [DAN](./configs/rec/dan/)                     | [AAAI 2020](https://arxiv.org/pdf/1912.10205)                                                    | ✅       | ✅       |                                             |
| [RobustScanner](./configs/rec/robustscanner/) | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3160_ECCV_2020_paper.php)     | ✅       | ✅       | [pretto0](https://github.com/pretto0)       |
| [AutoSTR](./configs/rec/autostr/)             | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690732.pdf)              | ✅       | ✅       |                                             |
| [SRN](./configs/rec/srn/)                     | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html) | ✅       | ✅       | [pretto0](https://github.com/pretto0)       |
| [SEED](./configs/rec/seed/)                   | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html) | ✅       | ✅       |                                             |
| [ABINet](./configs/rec/abinet/)               | [CVPR 2021](https://openaccess.thecvf.com//content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html) | ✅       | ✅       | [YesianRohn](https://github.com/YesianRohn) |
| [VisionLAN](./configs/rec/visionlan/)         | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html) | ✅       | ✅       | [YesianRohn](https://github.com/YesianRohn) |
| PIMNet                                        | [ACM MM 2021](https://dl.acm.org/doi/10.1145/3474085.3475238)                                    |          |          | TODO                                        |
| [SVTR](./configs/rec/svtrs/)                  | [IJCAI 2022](https://www.ijcai.org/proceedings/2022/124)                                         | ✅       | ✅       |                                             |
| [PARSeq](./configs/rec/parseq/)               | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)              | ✅       | ✅       |                                             |
| [MATRN](./configs/rec/matrn/)                 | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)              | ✅       | ✅       |                                             |
| [MGP-STR](./configs/rec/mgpstr/)              | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)              | ✅       | ✅       |                                             |
| [LPV](./configs/rec/lpv/)                     | [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0189.pdf)                                    | ✅       | ✅       |                                             |
| [MAERec](./configs/rec/maerec/)(Union14M)     | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf) | ✅       | ✅       |                                             |
| [LISTER](./configs/rec/lister/)               | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.pdf) | ✅       | ✅       |                                             |
| [CDistNet](./configs/rec/cdistnet/)           | [IJCV 2024](https://link.springer.com/article/10.1007/s11263-023-01880-0)                        | ✅       | ✅       | [YesianRohn](https://github.com/YesianRohn) |
| [BUSNet](./configs/rec/busnet/)               | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28402)                              | ✅       | ✅       |                                             |
| DCTC                                          | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28575)                              |          |          | TODO                                        |
| [CAM](./configs/rec/cam/)                     | [PR 2024](https://arxiv.org/abs/2402.13643)                                                      | ✅       | ✅       |                                             |
| [OTE](./configs/rec/ote/)                     | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.html) | ✅       | ✅       |                                             |
| CFF                                           | [IJCAI 2024](https://arxiv.org/abs/2407.05562)                                                   |          |          | TODO                                        |
| [DPTR](./configs/rec/dptr/)                   | [ACM MM 2024](https://arxiv.org/abs/2408.05706)                                                  |          |          | [fd-zs](https://github.com/fd-zs)           |
| VIPTR                                         | [ACM CIKM 2024](https://arxiv.org/abs/2401.10110)                                                |          |          | TODO                                        |
| [IGTR](./configs/rec/igtr/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10820836)                                      | ✅       | ✅       |                                             |
| [SMTR](./configs/rec/smtr/)                   | [AAAI 2025](https://arxiv.org/abs/2407.12317)                                                    | ✅       | ✅       |                                             |
| [CPPD](./configs/rec/cppd/)                   | [TPAMI Online Access](https://ieeexplore.ieee.org/document/10902187)                             | ✅       | ✅       |                                             |
| [FocalSVTR-CTC](./configs/rec/svtrs/)         | [2024](https://arxiv.org/abs/2407.12317)                                                         | ✅       | ✅       |                                             |
| [SVTRv2](./configs/rec/svtrv2/)               | [2024](https://arxiv.org/abs/2411.15858)                                                         | ✅       | ✅       |                                             |
| [ResNet+Trans-CTC](./configs/rec/svtrs/)      |                                                                                                  | ✅       | ✅       |                                             |
| [ViT-CTC](./configs/rec/svtrs/)               |                                                                                                  | ✅       | ✅       |                                             |

#### 核心贡献者

______________________________________________________________________

复旦大学[FVL实验室](https://fvl.fudan.edu.cn)的Yiming Lei ([pretto0](https://github.com/pretto0)), Xingsong Ye ([YesianRohn](https://github.com/YesianRohn)), and Shuai Zhao ([fd-zs](https://github.com/fd-zs))在Zhineng Chen老师([个人主页](https://zhinchenfd.github.io/))指导下完成了主要算法复现工作，感谢他们的贡献。

### 场景文本检测(STD)

开发中

### 端到端文本识别(Text Spotting)

开发中

______________________________________________________________________

## 引用

如果我们的工作对您的研究有所帮助，请引用：

```bibtex
@inproceedings{Du2024SVTRv2,
      title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
      author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
      booktitle={ICCV},
      year={2025}
}
```

## 致谢

本代码库基于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[PytorchOCR](https://github.com/WenmuZhou/PytorchOCR)和[MMOCR](https://github.com/open-mmlab/mmocr)构建，感谢他们的出色工作！

# OpenOCR

OpenOCR aims to establish a unified training and evaluation benchmark for scene text detection and recognition algorithms, at the same time, serves as the official code repository for the OCR team from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University.

We are actively developing and refining it and expect to release the first version in July 2024.

We sincerely welcome the researcher to recommend OCR or relevant algorithms and point out any potential factual errors or bugs. Upon receiving the suggestions, we will promptly evaluate and critically reproduce them. We look forward to collaborating with you to advance the development of OpenOCR and continuously contribute to the OCR community!

## Ours STR algorithms

  - [**IGTR**](./configs/rec/igtr/) (*Yongkun Du, Zhineng Chen\*, Yuchen Su, Caiyan Jia, Yu-Gang Jiang. Instruction-Guided Scene Text Recognition,* 2024. [Doc](./configs/rec/igtr/readme.md), [paper](https://arxiv.org/abs/2401.17851))
  - [**SVTRv2**](./configs/rec/svtrv2) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Yu-Gang Jiang. SVTRv2: Towards Arbitrary-Shaped Text Recognition with a Single Visual Model,* 2024. [paper coming soon]())
  - [**SMTR&FocalSVTR**](./configs/rec/smtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang. Out of Length Text Recognition with Sub-String Match,* 2024. [paper](https://arxiv.org/abs/2407.12317))
  - [**CDistNet**](./configs/rec/cdistnet/) (*Tianlun Zheng, Zhineng Chen\*, Shancheng Fang, Hongtao Xie, Yu-Gang Jiang. CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition,* IJCV 2024. [paper](https://link.springer.com/article/10.1007/s11263-023-01880-0))
  - [**CPPD**](./configs/rec/cppd/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Chenxia Li, Yuning Du, Yu-Gang Jiang. Context Perception Parallel Decoder for Scene Text Recognition,* 2023. [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_cppd.md), [paper](https://arxiv.org/abs/2307.12270))
  - [**SVTR**](./configs/rec/svtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model,* IJCAI 2022 (Long). [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_svtr.md), [paper](https://www.ijcai.org/proceedings/2022/124))
  - [**NRTR**](./configs/rec/nrtr/) (*Fenfen Sheng, Zhineng Chen\*, Bo Xu. NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition,* ICDAR 2019. [paper](https://arxiv.org/abs/1806.00926))

## STR

Reproduction schedule:

| Method             | Venue     | Training | Evaluation | Contributor |
|--------------------|-----------|----------|------------|-------------|
| CRNN               | [TPAMI2016](https://arxiv.org/abs/1507.05717) |    ✅    |     ✅     |             |
| [ASTER](./configs/rec/aster/)              |[TPAMI2018](https://ieeexplore.ieee.org/document/8395027)           |     ✅     |      ✅      | [pretto0](https://github.com/pretto0) |
| [NRTR](./configs/rec/nrtr/)               | [ICDAR2019](https://arxiv.org/abs/1806.00926) |    ✅    |     ✅     |             |
| [SAR](./configs/rec/sar/)                |     [AAAI2019](https://aaai.org/papers/08610-show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition/)      |     ✅     |      ✅      | [pretto0](https://github.com/pretto0) |
| [RobustScanner](./configs/rec/robustscanner/)      |      [ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3160_ECCV_2020_paper.php)     |     ✅     |      ✅      | [pretto0](https://github.com/pretto0) |
| [SRN](./configs/rec/srn/)                |  [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html) |     ✅     |       ✅     | [pretto0](https://github.com/pretto0) |
| [ABINet](./configs/rec/abinet/)             | [CVPR2021](https://openaccess.thecvf.com//content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html)   |    ✅    |      ✅    | [YesianRohn](https://github.com/YesianRohn) |
| [VisionLAN](./configs/rec/visionlan/)          | [ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html)  |    ✅    |    ✅      | [YesianRohn](https://github.com/YesianRohn) |
| [SVTR](./configs/rec/svtr/)               | [IJCAI2022](https://www.ijcai.org/proceedings/2022/124) |    ✅    |     ✅     |             |
| [PARSeq](./configs/rec/parseq/)             | [ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)  |    ✅    |     ✅     |             |
| MATRN              |  [ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)         |          |            |      TODO       |
| MGP-STR            |  [ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)         |          |            |      TODO       |
| [CPPD](./configs/rec/cppd/)               | [2023](https://arxiv.org/abs/2307.12270)      |    ✅    |     ✅     |             |
| [LPV](./configs/rec/lpv/)                | [IJCAI2023](https://www.ijcai.org/proceedings/2023/0189.pdf) |    ✅    |     ✅     |             |
| MAERec(Union14m)   | [ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf)  |        |        |     TODO      |
| [LISTER](./configs/rec/lister/)             | [ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.pdf)  |    ✅    |     ✅     |             |
| [CDistNet](./configs/rec/cdistnet/)           | [IJCV2024](https://link.springer.com/article/10.1007/s11263-023-01880-0)  |    ✅    |     ✅     | [YesianRohn](https://github.com/YesianRohn) |
| [IGTR](./configs/rec/igtr/)               | [2024](https://arxiv.org/abs/2401.17851)      |    ✅    |     ✅     |             |
| [SMTR](./configs/rec/smtr/)               | [2024](https://arxiv.org/abs/2407.12317)      |    ✅    |     ✅     |             |
| [FocalSVTR-CTC](./configs/rec/focalsvtr/)      | 2024      |    ✅    |     ✅     |             |
| [SVTRv2](./configs/rec/svtrv2/)             | 2024      |    ✅    |     ✅     |             |
| [ResNet+En-CTC](./configs/rec/svtr/)      |           |    ✅    |     ✅     |             |
| [ViT-CTC](./configs/rec/svtr/)            |           |    ✅    |     ✅     |             |

### Contributors
---

Yiming Lei ([pretto0](https://github.com/pretto0)) and Xingsong Ye ([YesianRohn](https://github.com/YesianRohn)) from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University, under the guidance of Professor Zhineng Chen, completed the majority of the algorithm reproduction work. Grateful for their outstanding contributions.

---
### 训练准备

- 修改数据集路径

```
Train:
  dataset:
    name: LMDBDataSet
    data_dir: Path to train data
...

Eval:
  dataset:
    name: LMDBDataSet
    data_dir: Path to eval data
```

- 启动训练

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrnet_ctc.yml
```

### 添加新算法

参考[提交PR流程](https://github.com/Topdu/OpenOCR/pull/2)

流程为：

1、先Fork OpenOCR 项目到自己Github仓库中。

2、git clone -b develop https://github.com/自己的用户名/OpenOCR.git （注意每次git clone 之前要保证自己的仓库是最新代码）。

3、参考svtrnet_ctc和svtr_base_cppd的代码结构，将新算法的preprocess、modeling.encoder、modeling.decoder、optimizer、loss、postprocess添加到代码中。

4、安装pre-commit，执行代码风格检查。

```
pip install pre-commit
pre-commit install
```

5、将新添加的算法训练、评估、测试跑通后，按照github提交commit的流程向源仓库提交PR。

## STD

## E2E


# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR). Thanks!

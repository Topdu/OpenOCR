<div align="center">

<h1> OpenOCR: An Open-Source Toolkit for General-OCR Research and Applications </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟. </h5>

<a href="https://github.com/Topdu/OpenOCR/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/Topdu/OpenOCR"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/Topdu/OpenOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/Topdu/OpenOCR?color=9ea"></a>
<a href="https://pepy.tech/project/openocr"><img src="https://static.pepy.tech/personalized-badge/openocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Clone%20downloads"></a>
<a href="https://github.com/Topdu/OpenOCR/stargazers"><img src="https://img.shields.io/github/stars/Topdu/OpenOCR?color=ccf"></a>
<a href="https://pypi.org/project/openocr-python/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openocr-python"></a>
<a href="https://pypi.org/project/openocr-python/"><img src="https://img.shields.io/pypi/dm/openocr-python?label=PyPI%20downloads"></a>

English | [简体中文](./README_ch.md)

</div>

______________________________________________________________________

OpenOCR is an open-source toolkit developed by the OCR team from [FVL Lab](https://fvl.fudan.edu.cn), Fudan University, under the guidance of Prof. [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ) and Prof. [Zhineng Chen](https://zhinchenfd.github.io). It focuses on 「General-OCR」 tasks, including **Text Detection and Recognition, Formula and Table Recognition**, as well as **Document Parsing and Understanding**. The toolkit integrates a unified training and evaluation benchmark, commercial-grade OCR and Document Parsing systems, and faithful reproductions of the core implementations from a wide range of academic papers.

OpenOCR aims to build a comprehensive open-source ecosystem for General-OCR, bridging academic research and real-world applications, and fostering the collaborative development and widespread deployment of OCR technologies across both research frontiers and industrial scenarios. We welcome researchers, developers, and industry partners to explore the toolkit and share feedback.

## Features

- 🔥**OpenDoc-0.1B: Ultra-Lightweight Document Parsing System with 0.1B Parameters**

  - ⚡\[[Quick Start](./docs/opendoc.md)\] [![HuggingFace](https://img.shields.io/badge/OpenDoc--0.1B-_Demo_on_HuggingFace-yellow?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/spaces/topdu/OpenDoc-0.1B-Demo)
    [![ModelScope](https://img.shields.io/badge/OpenDoc--0.1B-_Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://modelscope.cn/studios/topdktu/OpenDoc-0.1B-Demo) \[[Local Demo](./docs/opendoc.md#local-demo)\]

    - An ultra-lightweight document parsing system with only 0.1B parameters.
    - Two-stage pipeline:
      1. Layout analysis via **[PP-DocLayoutV2](https://www.paddleocr.ai/latest/version3.x/module_usage/layout_analysis.html)**.
      2. Unified recognition of text, formulas, and tables using the in-house model **[UniRec-0.1B](./docs/unirec.md)**
         - In the original version of **UniRec-0.1B**, only **text and formula recognition** were supported. In **OpenDoc-0.1B**, we **rebuilt UniRec-0.1B** to enable **unified recognition of text, formulas, and tables**.
    - Supports document parsing for **Chinese and English**.
    - Achieves **90.57% on [OmniDocBench (v1.5)](https://github.com/opendatalab/OmniDocBench/tree/main?tab=readme-ov-file#end-to-end-evaluation)**, outperforming many document parsing models based on multimodal large language models.

- 🔥**UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters**

  - \[[Doc](./docs/unirec.md)\] [![arXiv](https://img.shields.io/badge/UniRec--0.1B-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2512.21095) [![HuggingFace](https://img.shields.io/badge/UniRec--0.1B-_Demo_on_HuggingFace-yellow?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/spaces/topdu/OpenOCR-UniRec-Demo)
    [![ModelScope](https://img.shields.io/badge/UniRec--0.1B-_Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://modelscope.cn/studios/topdktu/OpenOCR-UniRec-Demo) \[[Local Demo](./docs/unirec.md#local-demo)\] \[[Hugging Face Model](https://huggingface.co/topdu/unirec-0.1b)\] \[[ModelScope Model](https://www.modelscope.cn/models/topdktu/unirec-0.1b)\]
    - Recognizing plain text (words, lines, paragraphs), formulas (single-line, multi-line), and mixed text-and-formulas content.
    - 0.1B parameters.
    - Trained from scratch on 40M data without pre-training.
    - Supporting both Chinese and English text/formulas recognition.

- 🔥**OpenOCR: A general OCR system with accuracy and efficiency**

  - ⚡\[[Quick Start](./docs/openocr.md#quick-start)\] [![HuggingFace](https://img.shields.io/badge/OpenOCR-_Demo_on_HuggingFace-yellow?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/spaces/topdu/OpenOCR-Demo)
    [![ModelScope](https://img.shields.io/badge/OpenOCR-_Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://modelscope.cn/studios/topdktu/OpenOCR-Demo) \[[Local Demo](./docs/openocr.md#local-demo)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
  - [Introduction](./docs/openocr.md)
    - A practical OCR system building on SVTRv2.
    - Outperforms [PP-OCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html) baseline by 4.5% on the [OCR competition leaderboard](https://aistudio.baidu.com/competition/detail/1131/0/leaderboard) in terms of accuracy, while preserving quite similar inference speed.
    - [x] Supports Chinese and English text detection and recognition.
    - [x] Provides server model and mobile model.
    - [x] Fine-tunes OpenOCR on a custom dataset: [Fine-tuning Det](./docs/finetune_det.md), [Fine-tuning Rec](./docs/finetune_rec.md).
    - [x] [ONNX model export for wider compatibility](./docs/openocr.md#export-onnx-model).

- 🔥**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition (ICCV 2025)**

  - \[[Doc](./configs/rec/svtrv2/)\] [![arXiv](https://img.shields.io/badge/SVTRv2-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.15858)  \[[Model](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Datasets](./docs/svtrv2.md#downloading-datasets)\] \[[Config, Training and Inference](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[Benchmark](./docs/svtrv2.md#results-benchmark--configs--checkpoints)\]
  - [Introduction](./docs/svtrv2.md)
    - A unified training and evaluation benchmark (on top of [Union14M](https://github.com/Mountchicken/Union14M?tab=readme-ov-file#3-union14m-dataset)) for Scene Text Recognition
    - Supports 24 Scene Text Recognition methods trained from scratch on the large-scale real dataset [Union14M-L-Filter](./docs/svtrv2.md#dataset-details), and will continue to add the latest methods.
    - Improves accuracy by 20-30% compared to models trained based on synthetic datasets.
    - Towards Arbitrary-Shaped Text Recognition and Language modeling with a Single Visual Model.
    - Surpasses Attention-based Encoder-Decoder Methods across challenging scenarios in terms of accuracy and speed
  - [Get Started](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch) with training a SOTA Scene Text Recognition model from scratch.

## Ours OCR algorithms

- [**UniRec-0.1B**](./configs/rec/unirec/) (*Yongkun Du, Zhineng Chen, Yazhen Xie, Weikang Bai, Hao Feng, Wei Shi, Yuchen Su, Can Huang, Yu-Gang Jiang. UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters,* Preprint. [Doc](./configs/rec/unirec/), [Paper](https://arxiv.org/pdf/2512.21095))
- [**MDiff4STR**](./configs/rec/mdiff4str/) (*Yongkun Du, Miaomiao Zhao, Songlin Fan, Zhineng Chen\*, Caiyan Jia, Yu-Gang Jiang. MDiff4STR: Mask Diffusion Model for Scene Text Recognition,* AAAI 2026 Oral. [Doc](./configs/rec/mdiff4str/), [Paper](https://arxiv.org/abs/2512.01422))
- **CMER** (*Weikang Bai, Yongkun Du, Yuchen Su, Yazhen Xie, Zhineng Chen\*. Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline,* AAAI 2026. [Paper](https://arxiv.org/abs/2512.13731), Code is coming soon.)
- **TextSSR** (*Xingsong Ye, Yongkun Du, Yunbo Tao, Zhineng Chen\*. TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition,* ICCV 2025. [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_TextSSR_Diffusion-based_Data_Synthesis_for_Scene_Text_Recognition_ICCV_2025_paper.pdf), [Code](https://github.com/YesianRohn/TextSSR))
- [**SVTRv2**](./configs/rec/svtrv2) (*Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition,* ICCV 2025. [Doc](./configs/rec/svtrv2/), [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html))
- [**IGTR**](./configs/rec/igtr/) (*Yongkun Du, Zhineng Chen\*, Yuchen Su, Caiyan Jia, Yu-Gang Jiang. Instruction-Guided Scene Text Recognition,* TPAMI 2025. [Doc](./configs/rec/igtr), [Paper](https://ieeexplore.ieee.org/document/10820836))
- [**CPPD**](./configs/rec/cppd/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Chenxia Li, Yuning Du, Yu-Gang Jiang. Context Perception Parallel Decoder for Scene Text Recognition,* TPAMI 2025. [PaddleOCR Doc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_cppd.en.md), [Paper](https://ieeexplore.ieee.org/document/10902187))
- [**SMTR&FocalSVTR**](./configs/rec/smtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xieping Gao, Yu-Gang Jiang. Out of Length Text Recognition with Sub-String Matching,* AAAI 2025. [Doc](./configs/rec/smtr/), [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32285))
- [**DPTR**](./configs/rec/dptr/) (*Shuai Zhao, Yongkun Du, Zhineng Chen\*, Yu-Gang Jiang. Decoder Pre-Training with only Text for Scene Text Recognition,* ACM MM 2024. [Paper](https://dl.acm.org/doi/10.1145/3664647.3681390))
- [**CDistNet**](./configs/rec/cdistnet/) (*Tianlun Zheng, Zhineng Chen\*, Shancheng Fang, Hongtao Xie, Yu-Gang Jiang. CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition,* IJCV 2024. [Paper](https://link.springer.com/article/10.1007/s11263-023-01880-0))
- **MRN** (*Tianlun Zheng, Zhineng Chen\*, Bingchen Huang, Wei Zhang, Yu-Gang Jiang. MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition,* ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_MRN_Multiplexed_Routing_Network_for_Incremental_Multilingual_Text_Recognition_ICCV_2023_paper.html), [Code](https://github.com/simplify23/MRN))
- **TPS++** (*Tianlun Zheng, Zhineng Chen\*, Jinfeng Bai, Hongtao Xie, Yu-Gang Jiang. TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition,* IJCAI 2023. [Paper](https://arxiv.org/abs/2305.05322), [Code](https://github.com/simplify23/TPS_PP))
- [**SVTR**](./configs/rec/svtr/) (*Yongkun Du, Zhineng Chen\*, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model,* IJCAI 2022 (Long). [PaddleOCR Doc](https://github.com/Topdu/PaddleOCR/blob/main/doc/doc_ch/algorithm_rec_svtr.md), [Paper](https://www.ijcai.org/proceedings/2022/124))
- [**NRTR**](./configs/rec/nrtr/) (*Fenfen Sheng, Zhineng Chen, Bo Xu. NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition,* ICDAR 2019. [Paper](https://arxiv.org/abs/1806.00926))

## Recent Updates

- **2025.12.25**: 🔥 Releasing [OpenDoc-0.1B](./docs/opendoc.md): Ultra-Lightweight Document Parsing System with 0.1B Parameters
- **2025.11.08**: Our paper [MDiff4STR](https://arxiv.org/abs/2512.01422) is accepted by AAAI 2026 (Oral). Accessible in [Doc](./configs/rec/mdiff4str/).
- **2025.11.08**: Our paper [CMER](https://arxiv.org/abs/2512.13731) is accepted by AAAI 2026. Code is coming soon.
- **2025.08.20**: 🔥 Releasing [UniRec-0.1B](https://arxiv.org/pdf/2512.21095): Unified Text and Formula Recognition with 0.1B Parameters
- **2025.07.10**: Our paper [SVTRv2](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html) is accepted by ICCV 2025. Accessible in [Doc](./configs/rec/svtrv2/).
- **2025.07.10**: Our paper [TextSSR](https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_TextSSR_Diffusion-based_Data_Synthesis_for_Scene_Text_Recognition_ICCV_2025_paper.pdf) is accepted by ICCV 2025. Accessible in [Code](https://github.com/YesianRohn/TextSSR).
- **2025.03.24**: 🔥 Releasing the feature of fine-tuning OpenOCR on a custom dataset: [Fine-tuning Det](./docs/finetune_det.md), [Fine-tuning Rec](./docs/finetune_rec.md)
- **2025.03.23**: 🔥 Releasing the feature of [ONNX model export for wider compatibility](#export-onnx-model).
- **2025.02.22**: Our paper [CPPD](https://ieeexplore.ieee.org/document/10902187) is accepted by TPAMI. Accessible in [Doc](./configs/rec/cppd/) and [PaddleOCR Doc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_cppd.en.md).
- **2024.12.31**: Our paper [IGTR](https://ieeexplore.ieee.org/document/10820836) is accepted by TPAMI. Accessible in [Doc](./configs/rec/igtr/).
- **2024.12.16**: Our paper [SMTR](https://ojs.aaai.org/index.php/AAAI/article/view/32285) is accepted by AAAI 2025. Accessible in [Doc](./configs/rec/smtr/).
- **2024.12.03**: The pre-training code for [DPTR](https://dl.acm.org/doi/10.1145/3664647.3681390) is merged.
- **🔥 2024.11.23 release notes**:
  - **OpenOCR: A general OCR system with accuracy and efficiency**
    - ⚡\[[Quick Start](./docs/openocr.md#quick-start)\] \[[Model](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)\] \[[ModelScope Demo](https://modelscope.cn/studios/topdktu/OpenOCR-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-Demo)\] \[[Local Demo](./docs/openocr.md#local-demo)\]  \[[PaddleOCR Implementation](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)\]
    - [Introduction](./docs/openocr.md)
  - **SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**
    - \[[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html)\] \[[Doc](./configs/rec/svtrv2/)\] \[[Model](./configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Datasets](./docs/svtrv2.md#downloading-datasets)\] \[[Config, Training and Inference](./configs/rec/svtrv2/readme.md#3-model-training--evaluation)\] \[[Benchmark](./docs/svtrv2.md#results--configs--checkpoints)\]
    - [Introduction](./docs/svtrv2.md)
    - [Get Started](./docs/svtrv2.md#get-started-with-training-a-sota-scene-text-recognition-model-from-scratch) with training a SOTA Scene Text Recognition model from scratch.

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
| [DPTR](./configs/rec/dptr/)                   | [ACM MM 2024](https://dl.acm.org/doi/10.1145/3664647.3681390)                                  |          |            | [fd-zs](https://github.com/fd-zs)           |
| VIPTR                                         | [ACM CIKM 2024](https://arxiv.org/abs/2401.10110)                                              |          |            | TODO                                        |
| [IGTR](./configs/rec/igtr/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10820836)                                    | ✅       | ✅         |                                             |
| [SMTR](./configs/rec/smtr/)                   | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32285)                            | ✅       | ✅         |                                             |
| [CPPD](./configs/rec/cppd/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10902187)                                    | ✅       | ✅         |                                             |
| [FocalSVTR-CTC](./configs/rec/svtrs/)         | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32285)                            | ✅       | ✅         |                                             |
| [SVTRv2](./configs/rec/svtrv2/)               | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html) | ✅       | ✅         |                                             |
| [ResNet+Trans-CTC](./configs/rec/svtrs/)      |                                                                                                | ✅       | ✅         |                                             |
| [ViT-CTC](./configs/rec/svtrs/)               |                                                                                                | ✅       | ✅         |                                             |
| [MDiff4STR](./configs/rec/mdiff4str/)         | [AAAI 2025 Oral](https://arxiv.org/abs/2512.01422)                                             | ✅       | ✅         |                                             |

### Scene Text Detection (STD)

TODO

### Text Spotting

TODO

______________________________________________________________________

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@inproceedings{Du2025SVTRv2,
  title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
  author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
  booktitle={ICCV},
  year={2025},
  pages={20147-20156}
}

@article{du2025unirec,
  title={UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters},
  author={Yongkun Du and Zhineng Chen and Yazhen Xie and Weikang Bai and Hao Feng and Wei Shi and Yuchen Su and Can Huang and Yu-Gang Jiang},
  journal={arXiv preprint arXiv:2512.21095},
  year={2025}
}
```

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!

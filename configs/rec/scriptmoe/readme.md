# ScriptMoE

- [ScriptMoE](#scriptmoe)
  - [1. Introduction](#1-introduction)
    - [1.1 Models and Results](#11-models-and-results)
  - [2. Environment](#2-environment)
  - [3. Model Training / Evaluation](#3-model-training--evaluation)
    - [Dataset Preparation](#dataset-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [Citation](#citation)

<a name="1"></a>

## 1. Introduction

Paper:

> **All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts** (Preprint, under review)
> Xingsong Ye, Yongkun Du, Jiaxin Zhang, Zhixian Li, Chong Sun, Chen Li, Jing Lyu, Lianwen Jin, Zhineng Chen\*
>
> Institute of Trustworthy Embodied AI, Fudan University \ Shanghai Key Laboratory of Multimodal Embodied AI \ WeChat Vision, Tencent Inc. \ South China University of Technology

Code: [YesianRohn/ScriptMoE](https://github.com/YesianRohn/ScriptMoE) & [Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)

Demo & Models: [MultilingualOCR-Demo](https://huggingface.co/spaces/Yesianrohn/MultilingualOCR-Demo)

<a name="model"></a>

Multilingual scene text recognition (STR) remains challenging due to the scarcity of training data for most languages and the difficulty of serving diverse scripts within a single model. Existing solutions either deploy one recognizer per language, inflating cost and introducing error accumulation, or rely on massive vision-language models (VLMs) that are expensive and still inaccurate on many scripts. In this work, we pursue an all-in-one multilingual recognizer that is simpler than per-language experts, lighter than VLMs, and more accurate than both. First, we construct **TextMuSS-10M**, a large-scale synthetic scene text dataset spanning 10 scripts and 229 languages. Second, we propose **ScriptMoE**, a script-aware Mixture-of-Experts (MoE) architecture. It shares a single visual encoder (SVTRv2) and replaces the dense decoder FFN with a sparse MoE block, in which an image-level router dispatches each image to the top-2 script-aligned experts and a shared expert absorbs cross-script knowledge. A lightweight four-way script-classification head supervises expert specialization. Extensive experiments on **TextMuSS-Bench** (10 scripts, 10,899 images) show that ScriptMoE achieves the highest accuracy of **82.06%**, outperforming the strongest STR baseline by 1.31%. On the CC-OCR end-to-end multilingual task, replacing only the recognizer in PP-OCRv5 with ScriptMoE lifts the F1 score from 65.71% to 80.89%, slightly surpassing the best VLM (80.73%) at a fraction of the parameter count.

### 1.1 Models and Results

- Model size and inference efficiency (measured on a single NVIDIA V100 with batch size 256):

|   Model   | Params (M) | Activated Params (M) | Latency (ms) | Throughput (img/s) | Mem (MB) |
| :-------: | :--------: | :------------------: | :----------: | :----------------: | :------: |
| ScriptMoE |   45.85    |        41.13         |    541.07    |       473.1        |  2091.0  |

- Test on **TextMuSS-Bench** (10 scripts, 10,899 real scene text images). Word accuracy (%) is reported; `Avg` is the arithmetic average across the ten scripts, and `/` indicates that the model does not support the script.

|        Method         | Arabic | Bangla | Chinese | Hindi | Japanese | Korean | Latin | Russian | Thai  | Tibetan | **Avg** |
| :-------------------: | :----: | :----: | :-----: | :---: | :------: | :----: | :---: | :-----: | :---: | :-----: | :-----: |
| **ScriptMoE (ours)**  |  78.09 |  88.30 |  95.38  | 86.77 |  71.21   | 87.19  | 91.67 |  61.20  | 72.00 |  88.76  | **82.06** |
|       SVTRv2-AR       |  75.11 |  87.79 |  94.15  | 87.28 |  69.36   | 85.86  | 92.27 |  59.30  | 69.60 |  86.80  |  80.75  |
|        MAERec         |  76.17 |  80.15 |  92.92  | 86.01 |  68.35   | 86.16  | 92.13 |  52.66  | 63.73 |  86.80  |  78.51  |
|       CDistNet        |  71.28 |  82.95 |  89.23  | 85.50 |  67.51   | 85.42  | 91.18 |  59.49  | 64.13 |  82.87  |  77.96  |
|        SVTRv2         |  70.85 |  83.46 |  93.23  | 84.73 |  66.67   | 85.86  | 91.52 |  47.91  | 66.27 |  85.39  |  77.59  |
|        PARSeq         |  67.87 |  78.12 |  87.08  | 83.21 |  65.15   | 84.98  | 90.81 |  61.01  | 62.67 |  84.55  |  76.55  |
|          SMTR         |  70.43 |  77.86 |  88.92  | 85.24 |  65.32   | 85.27  | 91.06 |  51.99  | 62.93 |  82.87  |  76.19  |
|       MDiff4STR       |  72.13 |  83.46 |  88.31  | 83.21 |  66.50   | 85.13  | 91.03 |  42.31  | 57.33 |  75.84  |  74.53  |
|          NRTR         |  67.66 |  79.64 |  81.23  | 83.21 |  60.44   | 81.15  | 89.97 |  58.53  | 59.39 |  71.07  |  73.23  |
|          SVTR         |  64.68 |  75.83 |  83.38  | 81.42 |  60.44   | 83.95  | 90.06 |  52.75  | 58.80 |  78.65  |  73.00  |
|          LPV          |  66.17 |  74.30 |  84.00  | 82.44 |  60.27   | 82.92  | 90.03 |  49.81  | 56.80 |  74.72  |  72.15  |
|        ABINet         |  61.06 |  69.47 |  82.15  | 79.90 |  58.08   | 81.74  | 88.84 |  50.76  | 53.20 |  73.88  |  69.91  |
|         CRNN          |  42.77 |  55.47 |  64.00  | 63.87 |  44.28   | 70.10  | 82.38 |  30.93  | 53.13 |  48.60  |  55.55  |
| *PP-OCRv5 MLT (zero-shot)* |  68.30 |   /    |  80.00  | 60.05 |  57.41   | 75.85  | 83.43 |  47.82  | 35.87 |    /    |  63.59  |
| *Qwen3.5-9B (zero-shot)* |  62.13 |  63.36 |  92.31  | 62.85 |  63.97   | 80.27  | 88.04 |  69.17  | 38.76 |  16.85  |  63.77  |

> Per-script size of TextMuSS-Bench: Arabic 470, Bangla 393, Chinese 325, Hindi 393, Japanese 594, Korean 679, Latin 5,885, Russian 1,054, Thai 750, Tibetan 356 (**Total 10,899**).

- End-to-end OCR on the **CC-OCR** multilingual task (F1 score, %). The detector (PP-OCRv5) is kept fixed and only the recognizer is replaced.

|             Method              | Korean | Japanese | Vietnamese | French | German | Italian | Spanish | Portuguese | Russian | Arabic | **Total** |
| :-----------------------------: | :----: | :------: | :--------: | :----: | :----: | :-----: | :-----: | :--------: | :-----: | :----: | :-------: |
| PP-OCRv5 Det + **ScriptMoE**    |  92.33 |  89.43   |   75.93    | 80.77  | 81.00  |  71.81  |  72.58  |   78.41    |  79.22  | 87.45  | **80.89** |
|          PP-OCRv5 MLT           |  78.58 |  76.13   |   33.67    | 64.86  | 62.30  |  69.47  |  68.48  |   72.03    |  49.67  | 81.93  |   65.71   |
|      *Qwen3.5-9B (zero-shot)*   |  79.03 |  75.17   |   81.43    | 82.83  | 76.82  |  78.77  |  85.08  |   87.24    |  78.97  | 82.01  |   80.73   |
|    *Gemini-1.5-Pro (zero-shot)* |  80.01 |  73.52   |   78.49    | 83.33  | 78.11  |  75.77  |  81.28  |   83.46    |  69.99  | 85.70  |   78.97   |
|        *GPT-4o (zero-shot)*     |  74.20 |  66.96   |   70.11    | 81.17  | 73.60  |  69.01  |  78.95  |   80.90    |  67.22  | 72.31  |   73.44   |
|            GoogleOCR            |  85.32 |  77.46   |   63.15    | 73.40  | 64.80  |  67.67  |  67.93  |   69.72    |  57.69  | 90.62  |   71.78   |

- Reference benchmarks on the dominant scripts, verifying that script-aware specialization does not trade off high-resource scripts.

**BCTR (Chinese)**

|   Model   | Document | Handwriting | Scene |  Web  | **Avg** |
| :-------: | :------: | :---------: | :---: | :---: | :-----: |
| ScriptMoE |  99.47   |    74.79    | 83.35 | 89.80 | **86.85** |
| SVTRv2-AR |  99.35   |    73.12    | 81.99 | 89.25 |  85.93  |

**Union14M-Benchmark (English)**

|   Model   | Curve | Multi-Oriented | Artistic | Contextless | Salient | Multi-Words | General | **Avg** |
| :-------: | :---: | :------------: | :------: | :---------: | :-----: | :---------: | :-----: | :-----: |
| ScriptMoE | 93.49 |     96.71      |  81.78   |    87.55    |  89.10  |    89.87    |  84.15  | **88.95** |
| SVTRv2-AR | 93.12 |     96.49      |  80.22   |    88.06    |  88.68  |    90.35    |  83.78  |  88.67  |

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

The training mixture is threefold:

| Data | Description |
| :--: | :---------- |
| **TextMuSS-10M** (synthetic) | 1M samples per script for **10 scripts**, covering **229 languages**. Built with the UnionST synthesis engine: 40% real lexicon words, 20% character-shuffled strings, 20% rare-character expansion strings, 20% News Crawl sentences; rendered on 8k SynthText backgrounds with curved / multi-directional / perspective effects. |
| **Union14M** (real) | Real English scene text data. |
| **BCTR** (real) | Real Chinese scene text data. |
| **MLT2019** (real) | A small amount of real multilingual data. |

The unified character set is obtained by de-duplicating and merging the per-script character tables (Arabic 747, Bangla 102, Chinese 16,147, Hindi 817, Japanese 4,398, Korean 3,687, Latin 923, Cyrillic 850, Thai 524, Tibetan 64) plus the space symbol, yielding **19,684 characters**. Together with BOS / EOS / PAD the decoder vocabulary has **19,687** entries. The dictionary shipped with this repo is [mlt_dict.txt](../../../tools/utils/mlt_dict.txt).

For evaluation, **TextMuSS-Bench** reuses the MLT2019 test split (seven scripts) and adds newly collected and annotated Russian, Thai and Tibetan images, giving 10,899 images in total.

Download: [TextMuSS-10M](https://huggingface.co/datasets/Yesianrohn/TextMuSS-10M) (training) and [TextMuSS-Bench](https://huggingface.co/datasets/Yesianrohn/TextMuSS-Bench) (evaluation).

After preparing the LMDB datasets, fill the paths into `data_dir_list` of the `Train` / `Eval` sections in [svtrv2_scriptmoe_mlt.yml](./svtrv2_scriptmoe_mlt.yml).

### Training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/train_rec.py --c configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml
```

Reference setup: 8 NVIDIA V100 (32GB) GPUs, AdamW with weight decay 0.05, peak learning rate 6.5e-4, global batch size 1024 (8 GPUs × 128 per GPU), one-cycle schedule with 1.5-epoch linear warm-up, 2 epochs in total (~92.7 GPU-hours, 11.8 hours wall-clock).

Note: ScriptMoE activates only `top_k` (2) of the 4 experts per sample, so `Global.find_unused_parameters` is set to `True` in the config to keep DistributedDataParallel working.

### Evaluation

```shell
python tools/eval_rec_all_mlt.py --c configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml
```

After a successful run, the results are saved in a csv file in `output_dir` in the config file.

### Inference

```shell
python tools/infer_rec.py --c configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@article{Ye2026ScriptMoE,
  title={All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts},
  author={Xingsong Ye and Yongkun Du and Jiaxin Zhang and Zhixian Li and Chong Sun and Chen Li and Jing Lyu and Lianwen Jin and Zhineng Chen},
  journal={arXiv preprint},
  year={2026}
}
```

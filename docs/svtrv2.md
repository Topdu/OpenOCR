# SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

\[[Paper](../configs/rec/svtrv2/SVTRv2.pdf)\] \[[Model](../configs/rec/svtrv2/readme.md#11-models-and-results)\] \[[Config, Training and Inference](../configs/rec/svtrv2/readme.md#3-model-training--evaluation)\]

## Introduction

Connectionist temporal classification (CTC)-based scene text recognition (STR) methods, e.g., SVTR, are widely employed in OCR applications, mainly due to their simple architecture, which only contains a visual model and a CTC-aligned linear classifier, and therefore fast inference. However, they generally have worse accuracy than encoder-decoder-based methods (EDTRs), particularly in challenging scenarios. In this paper, we propose SVTRv2, a CTC model that beats leading EDTRs in both accuracy and inference speed. SVTRv2 introduces novel upgrades to handle text irregularity and utilize linguistic context, which endows it with the capability to deal with challenging and diverse text instances. First, a multi-size resizing (MSR) strategy is proposed to adaptively resize the text and maintain its readability. Meanwhile, we introduce a feature rearrangement module (FRM) to ensure that visual features accommodate the alignment requirement of CTC well, thus alleviating the alignment puzzle. Second, we propose a semantic guidance module (SGM). It integrates linguistic context into the visual model, allowing it to leverage language information for improved accuracy. Moreover, SGM can be omitted at the inference stage and would not increase the inference cost. We evaluate SVTRv2 in both standard and recent challenging benchmarks, where SVTRv2 is fairly compared with 24 mainstream STR models across multiple scenarios, including different types of text irregularity, languages, and long text. The results indicate that SVTRv2 surpasses all the EDTRs across the scenarios in terms of accuracy and speed.

## A Unified Training and Evaluation Benchmark for Scene Text Recognition

Recent research shows that STR models' performance can be been significantly improved by leveraging large-scale real-world datasets. However, many previous methods are trained on synthetic datasets, which fail to reflect their performance in real-world scenarios. Additionally, recent approaches have used different real-world datasets and inconsistent evaluation protocols, making it difficult to compare their performance.

To this end, we established a unified benchmark to re-train and evaluate mainstream STR methods.

First, to evaluate the performance of STR methods across diverse scenarios, we selected Union14M-Benchmarks as the test set. This benchmark includes a variety of complex scenarios. Additionally, we reported results on six test sets (Common Benchmarks) used in previous studies. For the training set, we used the large-scale real-world training dataset Union14M-L. To avoid data leakage, we filtered out the overlapping samples between Union14M-L (training set) and Union14M-L-Benchmark (Test set), resulting in the Union14M-L-Filter training dataset.

Furthermore, previous methods used inconsistent hyperparameter settings during training, which contributed to variations in their performance. To ensure reliable evaluation, we standardized key settings that significantly affect accuracy, such as the number of training epochs, data augmentation strategies, input size, and evaluation strategies. This ensures the reliability of our results.

Subsequently, we trained 24 reproduced STR methods and SVTRv2 using the Union14M-L-Filter dataset and evaluated their performance on both Common Benchmarks and Union14M-L-Benchmark.

### Dataset Details

#### Training Dataset

- **Unified Training Set:** All models are trained from scratch on a unified dataset named **Union14M-L-Filter**. This dataset was derived from **Union14M-L**, with certain adjustments for filtering.
- **Composition of Datasets:** The training dataset includes different categories of samples, categorized as **Easy**, **Medium**, **Hard**, **Norm**, and **Challenging**.
  - **Union14M-L:** Contains 3,230,742 images in total.
  - **Union14M-L-Filter:** The filtered version contains 3,224,143 images.
  - **Overlap with Union14M-Benchmarks:** Only 6,599 images overlap between **Union14M-L** and the benchmark datasets used for evaluation.

|                       | Easy      | Medium  | Hard    | Norm    | Challenging | Total     |
| --------------------- | --------- | ------- | ------- | ------- | ----------- | --------- |
| **Union14M-L**        | 2,076,161 | 145,525 | 308,025 | 218,154 | 482,877     | 3,230,742 |
| **Union14M-L-Filter** | 2,073,822 | 144,677 | 306,771 | 217,070 | 481,803     | 3,224,143 |

#### Test Datasets

1. **Common Test Set:**
   Previous methods have primarily been evaluated on this set. While it includes some irregular text, the datasets are not highly challenging (e.g., little curva or rotation). Models trained on synthetic datasets often perform well here.
   - Includes 6 subsets with **regular** and **irregular** samples.
   - Example datasets: **IIIT5K**, **SVT**, **IC13**, **IC15**, **SVTP**, and **CUTE80**.

| Dataset    | Image Count | Characteristics                      |
| ---------- | ----------- | ------------------------------------ |
| **IIIT5K** | 3,000       | Regular                              |
| **SVT**    | 647         | Regular                              |
| **IC13**   | 857         | Regular                              |
| **IC15**   | 1,811       | Irregular (low resolution, blurring) |
| **SVTP**   | 645         | Irregular (affine, perspective)      |
| **CUTE80** | 288         | Irregular (curved, clear)            |

2. **Challenging Test Set (Union14M-L-Benchmark):**
   Introduced to test the full capabilities of STR (Scene Text Recognition) models. This set includes significantly more difficult text samples, featuring extreme curva, rotation, artistic styles, overlapping text, and other challenges.
   - Includes datasets such as **Curve**, **Multi-Oriented**, **Artistic**, **Contextless**, **Salient**, **Multi-word**, and **General**.

| Dataset            | Image Count | Characteristics                                   |
| ------------------ | ----------- | ------------------------------------------------- |
| **Curve**          | 2,426       | Severe curvature                                  |
| **Multi-Oriented** | 1,369       | Severe rotation, multi-angle, vertical directions |
| **Artistic**       | 900         | Artistic styles, often seen in logos              |
| **Contextless**    | 779         | No semantic meaning, out-of-dictionary words      |
| **Salient**        | 1,585       | Adjacent or overlapping text                      |
| **Multi-word**     | 829         | Contains multiple words                           |
| **General**        | 400,000     | Includes challenges like blurring, distortion     |

3. **Special Test Sets:**
   Designed to evaluate specific model capabilities beyond the above datasets.
   - **LTB (Long Text Benchmark):** Evaluates performance on long texts (25–36 characters), as models are typically trained on short texts (≤25 characters).
   - **OST (Occluded Scene Text):** Tests the model’s ability to infer text from damaged or partially erased samples.

| Dataset | Image Count | Characteristics                       |
| ------- | ----------- | ------------------------------------- |
| **LTB** | 3,376       | Long text (25 \< length \< 36)        |
| **OST** | 4832        | Partially erased/destroyed characters |

**Note**: Both **Union14M-L-Filter** and **Union14M-L-Benchmark** are based on [Union14M-L](https://github.com/Mountchicken/Union14M?tab=readme-ov-file#3-union14m-dataset) and therefore comply with its copyright. Additionally, Common Benchmarks and OST are derived from [PARSeq](https://github.com/baudm/parseq/tree/main?tab=readme-ov-file#datasets) and [VisionLAN](https://github.com/wangyuxin87/VisionLAN/blob/main/README.md#results-on-ost-datasets), respectively.

### Implementation Details

The optimal training hyperparameters for all models are usually not fixed. However, key setting such as **Training Epochs**, **Data Augmentation**, **Input Size**, **Data Type**, and **Evaluation Protocols**—which significantly impact accuracy—must be strictly standardized to ensure fair and unbiased comparisons of model performance. By following these standardizations, the results can accurately reflect the true capabilities of the models, unaffected by experimental inconsistencies. The specific setting include:

| **Setting**                 | **Detail**                                                                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training Set**            | For training, when the text length of a text image exceeds 25, samples with text length ≤ 25 are randomly selected from the training set to ensure models are only exposed to short texts (length ≤ 25). |
| **Test Sets**               | For all test sets except the long-text test set (LTB), text images with text length > 25 are filtered. Text length is calculated by removing spaces and non-94-character-set special characters. |
| **Input Size**              | Unless a method explicitly requires a dynamic size, models use a fixed input size of 32×128. If a model performs incorrectly with 32×128 during training, the original size is used. The test input size matches the training size. |
| **Data Augmentation**       | All models use the data augmentation strategy employed by PARSeq.                                                                                                             |
| **Training Epochs**         | Unless pre-training is required, all models are trained for 20 epochs.                                                                                                        |
| **Optimizer**               | AdamW is the default optimizer. If training fails to converge with AdamW, Adam or other optimizers are used.                                                                  |
| **Batch Size**              | Maximum batch size for all models is 1024. If single-GPU training is not feasible, 2 GPUs (512 per GPU) or 4 GPUs (256 per GPU) are used. If 4-GPU training out of memory, the batch size is halved, and the learning rate is adjusted accordingly. |
| **Learning Rate**           | Default learning rate for batch size 1024 is 0.00065. The learning rate is adjusted multiple times to achieve the best results.                                               |
| **Learning Rate Scheduler** | A linear warm-up for 1.5 epochs is followed by a OneCycle scheduler.                                                                                                          |
| **Weight Decay**            | Default weight decay is 0.05. NormLayer and Bias parameters have a weight decay of 0.                                                                                         |
| **Data Type**               | All models are trained with mixed precision.                                                                                                                                  |
| **EMA or Similar Tricks**   | No EMA or similar tricks are used for any model.                                                                                                                              |
| **Evaluation Protocols**    | Word accuracy is evaluated after filtering special characters and converting all text to lowercase.                                                                           |

## Get Started with training a SoTA Scene Text Recognition model from scratch.

### Installation

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR
# Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.8
conda activate openocr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Downloading Datasets

All data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/17mK_PLiBU0-a8qhUd5g4H6V03yUM2Gyp?usp=sharing).

#### The structure of Datasets and OpenOCR code will be organized as follows:

<details close>
  <summary><strong>Structure of Datasets and OpenOCR code</strong></summary>

````
```text
benchmark_bctr # Chinese text datasets, optional
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
iiit5k_test_images # for Latency Measurement, optional
ltb # Long Text Benchmark
OpenOCR
OST
synth # optional
├── MJ
│   ├── test
│   ├── train
│   └── val
└── ST
test # Common Benchmarks from PARSeq
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
u14m # lmdb format Union14M-Benchmark
├── artistic
├── contextless
├── curve
├── general
├── multi_oriented
├── multi_words
└── salient
Union14M-L-LMDB-Filtered # lmdb format Union14M-L-Filtered
├── train_challenging
├── train_easy
├── train_hard
├── train_medium
└── train_normal
```
````

</details>

#### **Datasets used during Training**

| Datsets           | Google Drive                                                                                             | Baidu Yun |
| ----------------- | -------------------------------------------------------------------------------------------------------- | --------- |
| Union14M-L-Filter | [LMDB archives](https://drive.google.com/drive/folders/1OlDWJZgvd6s4S09S3IGeAI90jI0i7AB_?usp=sharing)    |           |
| Evaluation        | [LMDB archives](https://drive.google.com/drive/folders/1EW0_YvmRSdpVOkR2guTQFrGz7KNqNc66?usp=drive_link) |           |

If you have downloaded Union14M-L, you can use [the filtered list of images](https://drive.google.com/drive/folders/1x1LC8C_W-Frl3sGV9i9_i_OD-bqNdodJ?usp=drive_link) to create an LMDB of the training set Union14M-L-Filter.

#### **Test Set**

| Datsets                   | Google Drive                                                                                             | Baidu Yun |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | --------- |
| Union14M-L-Benchmark      | [LMDB archives](https://drive.google.com/drive/folders/182RBQqjMVRZAXe0kpUJbMiDo7RWHwcSK?usp=drive_link) |           |
| Common-Benchmarks         | [LMDB archives](https://drive.google.com/drive/folders/103DA5Wu9YofDbE-JIUm5StJxTO2GixgS?usp=drive_link) |           |
| Long Text Benchmark (LTB) | [LMDB archives](https://drive.google.com/drive/folders/1XH4ADGir4KkeVrGpw0t8mDFBD0cKpQm_?usp=drive_link) |           |
| Occluded Scene Text (OST) | [LMDB archives](https://drive.google.com/drive/folders/1c0NVNF4yRACiP2IfeO_GnrHj0u_FhnpN?usp=drive_link) |           |

**Note**: Both **Union14M-L-Filter** and **Union14M-L-Benchmark** are based on [Union14M-L](https://github.com/Mountchicken/Union14M?tab=readme-ov-file#3-union14m-dataset) and therefore comply with its copyright. Common Benchmarks and OST are derived from [PARSeq](https://github.com/baudm/parseq/tree/main?tab=readme-ov-file#datasets) and [VisionLAN](https://github.com/wangyuxin87/VisionLAN/blob/main/README.md#results-on-ost-datasets), respectively.

### Training & Evaluation & Inference & Latency Measurement

**Note**: Take SVTRv2 as an example here. The execution commands for each model are listed in detail on their [readme pages](../configs/rec/).

#### Training

```shell
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_maxratio12.yml
# For Multi RTX 4090
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_maxratio12.yml
# 20epoch runs for about 6 hours
```

#### Evaluation

```shell
# short text: Common, Union14M-Benchmark, OST
python tools/eval_rec_all_ratio.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml
# long text
python tools/eval_rec_all_long_simple.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml
```

After a successful run, the results are saved in a csv file in `output_dir` in the config file.

#### Inference

```shell
python tools/infer_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

#### Latency Measurement

Firstly, downloading the IIIT5K images from [Google Drive](https://drive.google.com/drive/folders/1Po1LSBQb87DxGJuAgLNxhsJ-pdXxpIfS?usp=drive_link). Then, running the following command:

```shell
python tools/infer_rec.py --c configs/rec/svtrv2/svtrv2_rctc.yml --o Global.infer_img=../iiit5k_test_image
```

## Results & Configs & Checkpoints:

Downloading all model checkpoints from [Google Drive](<>) and [Baidu Yun](<>).

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
.tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-10jc{color:#F00;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-wn3b{background-color:#D2E4FC;text-align:center;vertical-align:middle}
.tg .tg-c4ze{color:#000000;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-0f79{background-color:#D2E4FC;color:#000000;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-s1e0{background-color:#D2E4FC;color:#F00;font-weight:bold;text-align:center;vertical-align:middle}
</style>

<table class="tg"><thead>
  <tr>
    <th class="tg-c4ze" colspan="2" rowspan="2">Method</th>
    <th class="tg-c4ze" rowspan="2">Venue</th>
    <th class="tg-c4ze" rowspan="2">Encoder</th>
    <th class="tg-c4ze" rowspan="2">Config</th>
    <th class="tg-c4ze" rowspan="2">Model</th>
    <th class="tg-c4ze" colspan="6">Common Benchmarks</th>
    <th class="tg-c4ze" rowspan="2">Avg</th>
    <th class="tg-c4ze" colspan="7">Union-14M Benchmarks</th>
    <th class="tg-c4ze" rowspan="2">Avg</th>
    <th class="tg-c4ze" rowspan="2">LTB</th>
    <th class="tg-c4ze" rowspan="2">OST</th>
    <th class="tg-c4ze" rowspan="2">Param<br> (M)</th>
    <th class="tg-c4ze" rowspan="2">Latency<br>(ms)</th>
  </tr>
  <tr>
    <th class="tg-0f79">IIIT</th>
    <th class="tg-0f79">SVT</th>
    <th class="tg-0f79">IC13</th>
    <th class="tg-0f79">IC15</th>
    <th class="tg-0f79">SVTP</th>
    <th class="tg-0f79">CUTE</th>
    <th class="tg-0f79">Curve</th>
    <th class="tg-0f79">Multi-Oriented</th>
    <th class="tg-0f79">Artistic</th>
    <th class="tg-0f79">Contextless</th>
    <th class="tg-0f79">Salient</th>
    <th class="tg-0f79">Multi-Words</th>
    <th class="tg-0f79">General</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="22">EDTRs</td>
    <td class="tg-nrix">ASTER</td>
    <td class="tg-nrix">TPAMI19</td>
    <td class="tg-nrix">ResNet31+LSTM</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">96.1 </td>
    <td class="tg-nrix">93.0 </td>
    <td class="tg-nrix">94.9 </td>
    <td class="tg-nrix">86.1 </td>
    <td class="tg-nrix">87.9 </td>
    <td class="tg-nrix">92.0 </td>
    <td class="tg-nrix">91.68 </td>
    <td class="tg-nrix">70.9 </td>
    <td class="tg-nrix">82.2 </td>
    <td class="tg-nrix">56.7 </td>
    <td class="tg-nrix">62.9 </td>
    <td class="tg-nrix">73.9 </td>
    <td class="tg-nrix">58.5 </td>
    <td class="tg-nrix">76.3 </td>
    <td class="tg-nrix">68.75 </td>
    <td class="tg-nrix">0.02 </td>
    <td class="tg-nrix">61.9 </td>
    <td class="tg-nrix">19.04 </td>
    <td class="tg-nrix">14.9 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">NRTR</td>
    <td class="tg-wn3b">ICDAR19</td>
    <td class="tg-wn3b">Stem+TF6</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.1 </td>
    <td class="tg-wn3b">96.8 </td>
    <td class="tg-wn3b">97.8 </td>
    <td class="tg-wn3b">88.9 </td>
    <td class="tg-wn3b">93.3 </td>
    <td class="tg-wn3b">94.4 </td>
    <td class="tg-wn3b">94.89 </td>
    <td class="tg-wn3b">67.9 </td>
    <td class="tg-wn3b">42.4 </td>
    <td class="tg-wn3b">66.5 </td>
    <td class="tg-wn3b">73.6 </td>
    <td class="tg-wn3b">66.4 </td>
    <td class="tg-wn3b">77.2 </td>
    <td class="tg-wn3b">78.3 </td>
    <td class="tg-wn3b">67.46 </td>
    <td class="tg-wn3b">2.00 </td>
    <td class="tg-wn3b">74.8 </td>
    <td class="tg-wn3b">44.26 </td>
    <td class="tg-wn3b">57.8</td>
  </tr>
  <tr>
    <td class="tg-nrix">MORAN</td>
    <td class="tg-nrix">PR19</td>
    <td class="tg-nrix">ResNet31+LSTM</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">96.7 </td>
    <td class="tg-nrix">91.7 </td>
    <td class="tg-nrix">94.6 </td>
    <td class="tg-nrix">84.6 </td>
    <td class="tg-nrix">85.7 </td>
    <td class="tg-nrix">90.3 </td>
    <td class="tg-nrix">90.61 </td>
    <td class="tg-nrix">51.2 </td>
    <td class="tg-nrix">15.5 </td>
    <td class="tg-nrix">51.3 </td>
    <td class="tg-nrix">61.2 </td>
    <td class="tg-nrix">43.2 </td>
    <td class="tg-nrix">64.1 </td>
    <td class="tg-nrix">69.3 </td>
    <td class="tg-nrix">50.82 </td>
    <td class="tg-nrix">0.06 </td>
    <td class="tg-nrix">57.9 </td>
    <td class="tg-nrix">17.35 </td>
    <td class="tg-nrix">16.8 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SAR</td>
    <td class="tg-wn3b">AAAI19</td>
    <td class="tg-wn3b">ResNet31+LSTM</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.1 </td>
    <td class="tg-wn3b">93.8 </td>
    <td class="tg-wn3b">96.7 </td>
    <td class="tg-wn3b">86.0 </td>
    <td class="tg-wn3b">87.9 </td>
    <td class="tg-wn3b">95.5 </td>
    <td class="tg-wn3b">93.01 </td>
    <td class="tg-wn3b">70.5 </td>
    <td class="tg-wn3b">51.8 </td>
    <td class="tg-wn3b">63.7 </td>
    <td class="tg-wn3b">73.9 </td>
    <td class="tg-wn3b">64.0 </td>
    <td class="tg-wn3b">79.1 </td>
    <td class="tg-wn3b">75.5 </td>
    <td class="tg-wn3b">68.36 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">60.6 </td>
    <td class="tg-wn3b">57.47 </td>
    <td class="tg-wn3b">63.1 </td>
  </tr>
  <tr>
    <td class="tg-nrix">DAN</td>
    <td class="tg-nrix">AAAI20</td>
    <td class="tg-nrix">ResNet45+FPN</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">97.5 </td>
    <td class="tg-nrix">94.7 </td>
    <td class="tg-nrix">96.5 </td>
    <td class="tg-nrix">87.1 </td>
    <td class="tg-nrix">89.1 </td>
    <td class="tg-nrix">94.4 </td>
    <td class="tg-nrix">93.24 </td>
    <td class="tg-nrix">74.9 </td>
    <td class="tg-nrix">63.3 </td>
    <td class="tg-nrix">63.4 </td>
    <td class="tg-nrix">70.6 </td>
    <td class="tg-nrix">70.2 </td>
    <td class="tg-nrix">71.1 </td>
    <td class="tg-nrix">76.8 </td>
    <td class="tg-nrix">70.05 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">61.8 </td>
    <td class="tg-nrix">27.71 </td>
    <td class="tg-nrix">10.1 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SRN</td>
    <td class="tg-wn3b">CVPR20</td>
    <td class="tg-wn3b">ResNet50+FPN</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">97.2 </td>
    <td class="tg-wn3b">96.3 </td>
    <td class="tg-wn3b">97.5 </td>
    <td class="tg-wn3b">87.9 </td>
    <td class="tg-wn3b">90.9 </td>
    <td class="tg-wn3b">96.9 </td>
    <td class="tg-wn3b">94.45 </td>
    <td class="tg-wn3b">78.1 </td>
    <td class="tg-wn3b">63.2 </td>
    <td class="tg-wn3b">66.3 </td>
    <td class="tg-wn3b">65.3 </td>
    <td class="tg-wn3b">71.4 </td>
    <td class="tg-wn3b">58.3 </td>
    <td class="tg-wn3b">76.5 </td>
    <td class="tg-wn3b">68.43 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">64.6 </td>
    <td class="tg-wn3b">51.70 </td>
    <td class="tg-wn3b">14.9 </td>
  </tr>
  <tr>
    <td class="tg-nrix">SEED</td>
    <td class="tg-nrix">CVPR20</td>
    <td class="tg-nrix">ResNet31+LSTM</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">96.5 </td>
    <td class="tg-nrix">93.2 </td>
    <td class="tg-nrix">94.2 </td>
    <td class="tg-nrix">87.5 </td>
    <td class="tg-nrix">88.7 </td>
    <td class="tg-nrix">93.4 </td>
    <td class="tg-nrix">92.24 </td>
    <td class="tg-nrix">69.1 </td>
    <td class="tg-nrix">80.9 </td>
    <td class="tg-nrix">56.9 </td>
    <td class="tg-nrix">63.9 </td>
    <td class="tg-nrix">73.4 </td>
    <td class="tg-nrix">61.3 </td>
    <td class="tg-nrix">76.5 </td>
    <td class="tg-nrix">68.87 </td>
    <td class="tg-nrix">0.10 </td>
    <td class="tg-nrix">62.6 </td>
    <td class="tg-nrix">23.95 </td>
    <td class="tg-nrix">15.3 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">AutoSTR</td>
    <td class="tg-wn3b">ECCV20</td>
    <td class="tg-wn3b">SearchCNN+LSTM</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">96.8 </td>
    <td class="tg-wn3b">92.4 </td>
    <td class="tg-wn3b">95.7 </td>
    <td class="tg-wn3b">86.6 </td>
    <td class="tg-wn3b">88.2 </td>
    <td class="tg-wn3b">93.4 </td>
    <td class="tg-wn3b">92.19 </td>
    <td class="tg-wn3b">72.1 </td>
    <td class="tg-wn3b">81.7 </td>
    <td class="tg-wn3b">56.7 </td>
    <td class="tg-wn3b">64.8 </td>
    <td class="tg-wn3b">75.4 </td>
    <td class="tg-wn3b">64.0 </td>
    <td class="tg-wn3b">75.9 </td>
    <td class="tg-wn3b">70.09 </td>
    <td class="tg-wn3b">0.10 </td>
    <td class="tg-wn3b">61.5 </td>
    <td class="tg-wn3b">6.04 </td>
    <td class="tg-wn3b">12.1 </td>
  </tr>
  <tr>
    <td class="tg-nrix">RoScanner</td>
    <td class="tg-nrix">ECCV20</td>
    <td class="tg-nrix">ResNet31</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.5 </td>
    <td class="tg-nrix">95.8 </td>
    <td class="tg-nrix">97.7 </td>
    <td class="tg-nrix">88.2 </td>
    <td class="tg-nrix">90.1 </td>
    <td class="tg-nrix">97.6 </td>
    <td class="tg-nrix">94.65 </td>
    <td class="tg-nrix">79.4 </td>
    <td class="tg-nrix">68.1 </td>
    <td class="tg-nrix">70.5 </td>
    <td class="tg-nrix">79.6 </td>
    <td class="tg-nrix">71.6 </td>
    <td class="tg-nrix">82.5 </td>
    <td class="tg-nrix">80.8 </td>
    <td class="tg-nrix">76.08 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">68.6 </td>
    <td class="tg-nrix">47.98 </td>
    <td class="tg-nrix">15.6 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">ABINet</td>
    <td class="tg-wn3b">CVPR21</td>
    <td class="tg-wn3b">ResNet45+TF3</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.5 </td>
    <td class="tg-wn3b">98.1 </td>
    <td class="tg-wn3b">97.7 </td>
    <td class="tg-wn3b">90.1 </td>
    <td class="tg-wn3b">94.1 </td>
    <td class="tg-wn3b">96.5 </td>
    <td class="tg-wn3b">95.83 </td>
    <td class="tg-wn3b">80.4 </td>
    <td class="tg-wn3b">69.0 </td>
    <td class="tg-wn3b">71.7 </td>
    <td class="tg-wn3b">74.7 </td>
    <td class="tg-wn3b">77.6 </td>
    <td class="tg-wn3b">76.8 </td>
    <td class="tg-wn3b">79.8 </td>
    <td class="tg-wn3b">75.72 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">75.0 </td>
    <td class="tg-wn3b">36.86 </td>
    <td class="tg-wn3b">13.7 </td>
  </tr>
  <tr>
    <td class="tg-nrix">VisionLAN</td>
    <td class="tg-nrix">ICCV21</td>
    <td class="tg-nrix">ResNet45+TF3</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.2 </td>
    <td class="tg-nrix">95.8 </td>
    <td class="tg-nrix">97.1 </td>
    <td class="tg-nrix">88.6 </td>
    <td class="tg-nrix">91.2 </td>
    <td class="tg-nrix">96.2 </td>
    <td class="tg-nrix">94.50 </td>
    <td class="tg-nrix">79.6 </td>
    <td class="tg-nrix">71.4 </td>
    <td class="tg-nrix">67.9 </td>
    <td class="tg-nrix">73.7 </td>
    <td class="tg-nrix">76.1 </td>
    <td class="tg-nrix">73.9 </td>
    <td class="tg-nrix">79.1 </td>
    <td class="tg-nrix">74.53 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">66.4 </td>
    <td class="tg-nrix">32.88 </td>
    <td class="tg-nrix">10.7 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">PARSeq</td>
    <td class="tg-wn3b">ECCV22</td>
    <td class="tg-wn3b">ViT-S</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.9 </td>
    <td class="tg-wn3b">98.1 </td>
    <td class="tg-wn3b">98.4 </td>
    <td class="tg-wn3b">90.1 </td>
    <td class="tg-wn3b">94.3 </td>
    <td class="tg-wn3b">98.6 </td>
    <td class="tg-wn3b">96.40 </td>
    <td class="tg-wn3b">87.6 </td>
    <td class="tg-wn3b">88.8 </td>
    <td class="tg-wn3b">76.5 </td>
    <td class="tg-wn3b">83.4 </td>
    <td class="tg-wn3b">84.4 </td>
    <td class="tg-wn3b">84.3 </td>
    <td class="tg-wn3b">84.9 </td>
    <td class="tg-wn3b">84.26 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">79.9 </td>
    <td class="tg-wn3b">23.83 </td>
    <td class="tg-wn3b">19.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">MATRN</td>
    <td class="tg-nrix">ECCV22</td>
    <td class="tg-nrix">ResNet45+TF3</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.8 </td>
    <td class="tg-10jc">98.3 </td>
    <td class="tg-nrix">97.9 </td>
    <td class="tg-nrix">90.3 </td>
    <td class="tg-10jc">95.2 </td>
    <td class="tg-nrix">97.2 </td>
    <td class="tg-nrix">96.29 </td>
    <td class="tg-nrix">82.2 </td>
    <td class="tg-nrix">73.0 </td>
    <td class="tg-nrix">73.4 </td>
    <td class="tg-nrix">76.9 </td>
    <td class="tg-nrix">79.4 </td>
    <td class="tg-nrix">77.4 </td>
    <td class="tg-nrix">81.0 </td>
    <td class="tg-nrix">77.62 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">77.8 </td>
    <td class="tg-nrix">44.34 </td>
    <td class="tg-nrix">21.3 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">MGP-STR</td>
    <td class="tg-wn3b">ECCV22</td>
    <td class="tg-wn3b">ViT-B</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">97.9 </td>
    <td class="tg-wn3b">97.8 </td>
    <td class="tg-wn3b">97.1 </td>
    <td class="tg-wn3b">89.6 </td>
    <td class="tg-wn3b">95.2 </td>
    <td class="tg-wn3b">96.9 </td>
    <td class="tg-wn3b">95.74 </td>
    <td class="tg-wn3b">85.2 </td>
    <td class="tg-wn3b">83.7 </td>
    <td class="tg-wn3b">72.6 </td>
    <td class="tg-wn3b">75.1 </td>
    <td class="tg-wn3b">79.8 </td>
    <td class="tg-wn3b">71.1 </td>
    <td class="tg-wn3b">83.1 </td>
    <td class="tg-wn3b">78.65 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">78.6 </td>
    <td class="tg-wn3b">148.00 </td>
    <td class="tg-wn3b">8.2 </td>
  </tr>
  <tr>
    <td class="tg-nrix">CPPD-B</td>
    <td class="tg-nrix">Preprint</td>
    <td class="tg-nrix">SVTR-B</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">99.0 </td>
    <td class="tg-nrix">97.8 </td>
    <td class="tg-nrix">98.2 </td>
    <td class="tg-nrix">90.4 </td>
    <td class="tg-nrix">94.0 </td>
    <td class="tg-10jc">99.0 </td>
    <td class="tg-nrix">96.40 </td>
    <td class="tg-nrix">86.2 </td>
    <td class="tg-nrix">78.7 </td>
    <td class="tg-nrix">76.5 </td>
    <td class="tg-nrix">82.9 </td>
    <td class="tg-nrix">83.5 </td>
    <td class="tg-nrix">81.9 </td>
    <td class="tg-nrix">83.5 </td>
    <td class="tg-nrix">81.91 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">79.6 </td>
    <td class="tg-nrix">27.00 </td>
    <td class="tg-nrix">8.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">LPV-B</td>
    <td class="tg-wn3b">IJCAI23</td>
    <td class="tg-wn3b">SVTR-B</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.6 </td>
    <td class="tg-wn3b">97.8 </td>
    <td class="tg-wn3b">98.1 </td>
    <td class="tg-wn3b">89.8 </td>
    <td class="tg-wn3b">93.6 </td>
    <td class="tg-wn3b">97.6 </td>
    <td class="tg-wn3b">95.93 </td>
    <td class="tg-wn3b">86.2 </td>
    <td class="tg-wn3b">78.7 </td>
    <td class="tg-wn3b">75.8 </td>
    <td class="tg-wn3b">80.2 </td>
    <td class="tg-wn3b">82.9 </td>
    <td class="tg-wn3b">81.6 </td>
    <td class="tg-wn3b">82.9 </td>
    <td class="tg-wn3b">81.20 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">77.7 </td>
    <td class="tg-wn3b">30.54 </td>
    <td class="tg-wn3b">12.1 </td>
  </tr>
  <tr>
    <td class="tg-nrix">MAERec</td>
    <td class="tg-nrix">ICCV23</td>
    <td class="tg-nrix">ViT-S</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-10jc">99.2 </td>
    <td class="tg-nrix">97.8 </td>
    <td class="tg-nrix">98.2 </td>
    <td class="tg-nrix">90.4 </td>
    <td class="tg-nrix">94.3 </td>
    <td class="tg-nrix">98.3 </td>
    <td class="tg-nrix">96.36 </td>
    <td class="tg-nrix">89.1 </td>
    <td class="tg-nrix">87.1 </td>
    <td class="tg-nrix">79.0 </td>
    <td class="tg-nrix">84.2 </td>
    <td class="tg-10jc">86.3 </td>
    <td class="tg-nrix">85.9 </td>
    <td class="tg-nrix">84.6 </td>
    <td class="tg-nrix">85.17 </td>
    <td class="tg-nrix">9.80 </td>
    <td class="tg-nrix">76.4 </td>
    <td class="tg-nrix">35.69 </td>
    <td class="tg-nrix">58.4 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">LISTER</td>
    <td class="tg-wn3b">ICCV23</td>
    <td class="tg-wn3b">FocalNet-B</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.8</td>
    <td class="tg-wn3b">97.5</td>
    <td class="tg-wn3b">98.6</td>
    <td class="tg-wn3b">90.0 </td>
    <td class="tg-wn3b">94.4</td>
    <td class="tg-wn3b">96.9</td>
    <td class="tg-wn3b">95.48 </td>
    <td class="tg-wn3b">78.7</td>
    <td class="tg-wn3b">68.8</td>
    <td class="tg-wn3b">73.7</td>
    <td class="tg-wn3b">81.6</td>
    <td class="tg-wn3b">74.8</td>
    <td class="tg-wn3b">82.4</td>
    <td class="tg-wn3b">83.5</td>
    <td class="tg-wn3b">77.64 </td>
    <td class="tg-wn3b">36.3</td>
    <td class="tg-wn3b">77.1</td>
    <td class="tg-wn3b">51.11</td>
    <td class="tg-wn3b">20.4</td>
  </tr>
  <tr>
    <td class="tg-nrix">CDistNet</td>
    <td class="tg-nrix">IJCV24</td>
    <td class="tg-nrix">ResNet45+TF3</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.7 </td>
    <td class="tg-nrix">97.1 </td>
    <td class="tg-nrix">97.8 </td>
    <td class="tg-nrix">89.6 </td>
    <td class="tg-nrix">93.5 </td>
    <td class="tg-nrix">96.9 </td>
    <td class="tg-nrix">95.59 </td>
    <td class="tg-nrix">81.7 </td>
    <td class="tg-nrix">77.1 </td>
    <td class="tg-nrix">72.6 </td>
    <td class="tg-nrix">78.2 </td>
    <td class="tg-nrix">79.9 </td>
    <td class="tg-nrix">79.7 </td>
    <td class="tg-nrix">81.1 </td>
    <td class="tg-nrix">78.62 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">71.8 </td>
    <td class="tg-nrix">43.32 </td>
    <td class="tg-nrix">62.9 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">CAM</td>
    <td class="tg-wn3b">PR24</td>
    <td class="tg-wn3b">ConvNeXtV2-T</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.2 </td>
    <td class="tg-wn3b">96.1 </td>
    <td class="tg-wn3b">96.6 </td>
    <td class="tg-wn3b">89.0 </td>
    <td class="tg-wn3b">93.5 </td>
    <td class="tg-wn3b">96.2 </td>
    <td class="tg-wn3b">94.94 </td>
    <td class="tg-wn3b">85.4 </td>
    <td class="tg-s1e0">89.0 </td>
    <td class="tg-wn3b">72.0 </td>
    <td class="tg-wn3b">75.4 </td>
    <td class="tg-wn3b">84.0 </td>
    <td class="tg-wn3b">74.8 </td>
    <td class="tg-wn3b">83.1 </td>
    <td class="tg-wn3b">80.52 </td>
    <td class="tg-wn3b">0.52 </td>
    <td class="tg-wn3b">74.2 </td>
    <td class="tg-wn3b">58.66 </td>
    <td class="tg-wn3b">35.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">BUSNet</td>
    <td class="tg-nrix">AAAI24</td>
    <td class="tg-nrix">ViT-S</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.3 </td>
    <td class="tg-nrix">98.1 </td>
    <td class="tg-nrix">97.8 </td>
    <td class="tg-nrix">90.2 </td>
    <td class="tg-nrix">95.3 </td>
    <td class="tg-nrix">96.5 </td>
    <td class="tg-nrix">96.06 </td>
    <td class="tg-nrix">83.0 </td>
    <td class="tg-nrix">82.3 </td>
    <td class="tg-nrix">70.8 </td>
    <td class="tg-nrix">77.9 </td>
    <td class="tg-nrix">78.8 </td>
    <td class="tg-nrix">71.2 </td>
    <td class="tg-nrix">82.6 </td>
    <td class="tg-nrix">78.10 </td>
    <td class="tg-nrix">0.00 </td>
    <td class="tg-nrix">78.7 </td>
    <td class="tg-nrix">32.10 </td>
    <td class="tg-nrix">12.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">OTE</td>
    <td class="tg-wn3b">CVPR24</td>
    <td class="tg-wn3b">SVTR-B</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.6 </td>
    <td class="tg-wn3b">96.6 </td>
    <td class="tg-wn3b">98.0 </td>
    <td class="tg-wn3b">90.1 </td>
    <td class="tg-wn3b">94.0 </td>
    <td class="tg-wn3b">97.2 </td>
    <td class="tg-wn3b">95.74 </td>
    <td class="tg-wn3b">86.0 </td>
    <td class="tg-wn3b">75.8 </td>
    <td class="tg-wn3b">74.6 </td>
    <td class="tg-wn3b">74.7 </td>
    <td class="tg-wn3b">81.0 </td>
    <td class="tg-wn3b">65.3 </td>
    <td class="tg-wn3b">82.3 </td>
    <td class="tg-wn3b">77.09 </td>
    <td class="tg-wn3b">0.00 </td>
    <td class="tg-wn3b">77.8 </td>
    <td class="tg-wn3b">20.28 </td>
    <td class="tg-wn3b">18.1 </td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="5">CTCs</td>
    <td class="tg-nrix">CRNN</td>
    <td class="tg-nrix">TPAMI16</td>
    <td class="tg-nrix">ResNet31+LSTM</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">95.8 </td>
    <td class="tg-nrix">91.8 </td>
    <td class="tg-nrix">94.6 </td>
    <td class="tg-nrix">84.9 </td>
    <td class="tg-nrix">83.1 </td>
    <td class="tg-nrix">91.0 </td>
    <td class="tg-nrix">90.21 </td>
    <td class="tg-nrix">48.1 </td>
    <td class="tg-nrix">13.0 </td>
    <td class="tg-nrix">51.2 </td>
    <td class="tg-nrix">62.3 </td>
    <td class="tg-nrix">41.4 </td>
    <td class="tg-nrix">60.4 </td>
    <td class="tg-nrix">68.2 </td>
    <td class="tg-nrix">49.24 </td>
    <td class="tg-nrix">47.21 </td>
    <td class="tg-nrix">58.0 </td>
    <td class="tg-nrix">16.20 </td>
    <td class="tg-nrix">5.8 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SVTR</td>
    <td class="tg-wn3b">IJCAI22</td>
    <td class="tg-wn3b">SVTR-B</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">98.0 </td>
    <td class="tg-wn3b">97.1 </td>
    <td class="tg-wn3b">97.3 </td>
    <td class="tg-wn3b">88.6 </td>
    <td class="tg-wn3b">90.7 </td>
    <td class="tg-wn3b">95.8 </td>
    <td class="tg-wn3b">94.58 </td>
    <td class="tg-wn3b">76.2 </td>
    <td class="tg-wn3b">44.5 </td>
    <td class="tg-wn3b">67.8 </td>
    <td class="tg-wn3b">78.7 </td>
    <td class="tg-wn3b">75.2 </td>
    <td class="tg-wn3b">77.9 </td>
    <td class="tg-wn3b">77.8 </td>
    <td class="tg-wn3b">71.17 </td>
    <td class="tg-wn3b">45.08 </td>
    <td class="tg-wn3b">69.6 </td>
    <td class="tg-wn3b">18.09 </td>
    <td class="tg-wn3b">6.2 </td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">SVTRv2</td>
    <td class="tg-nrix" rowspan="3">Preprint</td>
    <td class="tg-nrix">SVTRv2-T</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-nrix">98.6 </td>
    <td class="tg-nrix">96.6 </td>
    <td class="tg-nrix">98.0 </td>
    <td class="tg-nrix">88.4 </td>
    <td class="tg-nrix">90.5 </td>
    <td class="tg-nrix">96.5 </td>
    <td class="tg-nrix">94.78 </td>
    <td class="tg-nrix">83.6 </td>
    <td class="tg-nrix">76.0 </td>
    <td class="tg-nrix">71.2 </td>
    <td class="tg-nrix">82.4 </td>
    <td class="tg-nrix">77.2 </td>
    <td class="tg-nrix">82.3 </td>
    <td class="tg-nrix">80.7 </td>
    <td class="tg-nrix">79.05 </td>
    <td class="tg-nrix">47.83 </td>
    <td class="tg-nrix">71.4 </td>
    <td class="tg-nrix">5.13 </td>
    <td class="tg-nrix">5.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SVTRv2-S</td>
    <td class="tg-wn3b">Config</td>
    <td class="tg-wn3b">ckpt</td>
    <td class="tg-wn3b">99.0 </td>
    <td class="tg-s1e0">98.3 </td>
    <td class="tg-wn3b">98.5 </td>
    <td class="tg-wn3b">89.5 </td>
    <td class="tg-wn3b">92.9 </td>
    <td class="tg-wn3b">98.6 </td>
    <td class="tg-wn3b">96.13 </td>
    <td class="tg-wn3b">88.3 </td>
    <td class="tg-wn3b">84.6 </td>
    <td class="tg-wn3b">76.5 </td>
    <td class="tg-wn3b">84.3 </td>
    <td class="tg-wn3b">83.3 </td>
    <td class="tg-wn3b">85.4 </td>
    <td class="tg-wn3b">83.5 </td>
    <td class="tg-wn3b">83.70 </td>
    <td class="tg-wn3b">47.57 </td>
    <td class="tg-wn3b">78.0 </td>
    <td class="tg-wn3b">11.25 </td>
    <td class="tg-wn3b">5.3</td>
  </tr>
  <tr>
    <td class="tg-nrix">SVTRv2-B</td>
    <td class="tg-nrix">Config</td>
    <td class="tg-nrix">ckpt</td>
    <td class="tg-10jc">99.2 </td>
    <td class="tg-nrix">98.0 </td>
    <td class="tg-10jc">98.7 </td>
    <td class="tg-10jc">91.1 </td>
    <td class="tg-nrix">93.5 </td>
    <td class="tg-10jc">99.0 </td>
    <td class="tg-10jc">96.57 </td>
    <td class="tg-10jc">90.6 </td>
    <td class="tg-10jc">89.0 </td>
    <td class="tg-10jc">79.3 </td>
    <td class="tg-10jc">86.1 </td>
    <td class="tg-nrix">86.2 </td>
    <td class="tg-10jc">86.7 </td>
    <td class="tg-10jc">85.1 </td>
    <td class="tg-10jc">86.14 </td>
    <td class="tg-10jc">50.23 </td>
    <td class="tg-10jc">80.0 </td>
    <td class="tg-nrix">19.76 </td>
    <td class="tg-nrix">7.0 </td>
  </tr>
</tbody></table>

**Note**: TF$\_n$ denotes the $n$-layer Transformer block. $Size$ denotes the number of parameters ($M$). $Latency$ is measured on one NVIDIA 1080Ti GPU with Pytorch Dynamic mode.

## Results when trained on synthetic datasets ($ST$ + $MJ$).

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
.tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:0px;color:#444;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:0px;color:#fff;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-10jc{color:#F00;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-wn3b{background-color:#D2E4FC;text-align:center;vertical-align:middle}
.tg .tg-c4ze{color:#000000;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-sjbb{background-color:#D2E4FC;color:#F00;text-align:center;vertical-align:middle}
.tg .tg-0f79{background-color:#D2E4FC;color:#000000;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-s1e0{background-color:#D2E4FC;color:#F00;font-weight:bold;text-align:center;vertical-align:middle}
</style>

<table class="tg"><thead>
  <tr>
    <th class="tg-c4ze" colspan="2" rowspan="2">Method</th>
    <th class="tg-c4ze" rowspan="2">Venue</th>
    <th class="tg-c4ze" rowspan="2">Encoder</th>
    <th class="tg-c4ze" colspan="6">Common Benchmarks</th>
    <th class="tg-c4ze" rowspan="2">Avg</th>
    <th class="tg-c4ze" colspan="7">Union-14M Benchmarks</th>
    <th class="tg-c4ze" rowspan="2">Avg</th>
    <th class="tg-c4ze" rowspan="2">Param<br>(M)</th>
  </tr>
  <tr>
    <th class="tg-0f79">IC13</th>
    <th class="tg-0f79">SVT</th>
    <th class="tg-0f79">IIIT</th>
    <th class="tg-0f79">IC15</th>
    <th class="tg-0f79">SVTP</th>
    <th class="tg-0f79">CUTE</th>
    <th class="tg-0f79">Curve</th>
    <th class="tg-0f79">Multi-Oriented</th>
    <th class="tg-0f79">Artistic</th>
    <th class="tg-0f79">Contextless</th>
    <th class="tg-0f79">Salient</th>
    <th class="tg-0f79">Multi-Words</th>
    <th class="tg-0f79">General</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="27">EDTRs</td>
    <td class="tg-nrix">ASTER</td>
    <td class="tg-nrix">TPAMI19</td>
    <td class="tg-nrix">ResNet+LSTM</td>
    <td class="tg-nrix">93.3 </td>
    <td class="tg-nrix">90.0 </td>
    <td class="tg-nrix">90.8 </td>
    <td class="tg-nrix">74.7 </td>
    <td class="tg-nrix">80.2 </td>
    <td class="tg-nrix">80.9</td>
    <td class="tg-nrix">84.98</td>
    <td class="tg-nrix">34.0 </td>
    <td class="tg-nrix">10.2 </td>
    <td class="tg-nrix">27.7 </td>
    <td class="tg-nrix">33.0 </td>
    <td class="tg-nrix">48.2 </td>
    <td class="tg-nrix">27.6 </td>
    <td class="tg-nrix">39.8 </td>
    <td class="tg-nrix">31.50 </td>
    <td class="tg-nrix">27.2 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">NRTR</td>
    <td class="tg-wn3b">ICDAR19</td>
    <td class="tg-wn3b">Stem+TF6</td>
    <td class="tg-wn3b">90.1 </td>
    <td class="tg-wn3b">91.5 </td>
    <td class="tg-wn3b">95.8 </td>
    <td class="tg-wn3b">79.4 </td>
    <td class="tg-wn3b">86.6 </td>
    <td class="tg-wn3b">80.9</td>
    <td class="tg-wn3b">87.38</td>
    <td class="tg-wn3b">31.7 </td>
    <td class="tg-wn3b">4.4 </td>
    <td class="tg-wn3b">36.6 </td>
    <td class="tg-wn3b">37.3 </td>
    <td class="tg-wn3b">30.6 </td>
    <td class="tg-wn3b">54.9 </td>
    <td class="tg-wn3b">48.0 </td>
    <td class="tg-wn3b">34.79</td>
    <td class="tg-wn3b">31.7 </td>
  </tr>
  <tr>
    <td class="tg-nrix">MORAN</td>
    <td class="tg-nrix">PR19</td>
    <td class="tg-nrix">ResNet+LSTM</td>
    <td class="tg-nrix">91.0 </td>
    <td class="tg-nrix">83.9 </td>
    <td class="tg-nrix">91.3 </td>
    <td class="tg-nrix">68.4 </td>
    <td class="tg-nrix">73.3 </td>
    <td class="tg-nrix">75.7</td>
    <td class="tg-nrix">80.60 </td>
    <td class="tg-nrix">8.9 </td>
    <td class="tg-nrix">0.7 </td>
    <td class="tg-nrix">29.4 </td>
    <td class="tg-nrix">20.7 </td>
    <td class="tg-nrix">17.9 </td>
    <td class="tg-nrix">23.8 </td>
    <td class="tg-nrix">35.2 </td>
    <td class="tg-nrix">19.51</td>
    <td class="tg-nrix">17.4 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SAR</td>
    <td class="tg-wn3b">AAAI19</td>
    <td class="tg-wn3b">ResNet+LSTM</td>
    <td class="tg-wn3b">91.5 </td>
    <td class="tg-wn3b">84.5 </td>
    <td class="tg-wn3b">91.0 </td>
    <td class="tg-wn3b">69.2 </td>
    <td class="tg-wn3b">76.4 </td>
    <td class="tg-wn3b">83.5</td>
    <td class="tg-wn3b">82.68</td>
    <td class="tg-wn3b">44.3 </td>
    <td class="tg-wn3b">7.7 </td>
    <td class="tg-wn3b">42.6 </td>
    <td class="tg-wn3b">44.2 </td>
    <td class="tg-wn3b">44.0 </td>
    <td class="tg-wn3b">51.2 </td>
    <td class="tg-wn3b">50.5 </td>
    <td class="tg-wn3b">40.64</td>
    <td class="tg-wn3b">57.7 </td>
  </tr>
  <tr>
    <td class="tg-nrix">DAN</td>
    <td class="tg-nrix">AAAI20</td>
    <td class="tg-nrix">ResNet+FPN</td>
    <td class="tg-nrix">93.4 </td>
    <td class="tg-nrix">87.5 </td>
    <td class="tg-nrix">92.1 </td>
    <td class="tg-nrix">71.6 </td>
    <td class="tg-nrix">78.0 </td>
    <td class="tg-nrix">81.3</td>
    <td class="tg-nrix">83.98</td>
    <td class="tg-nrix">26.7 </td>
    <td class="tg-nrix">1.5 </td>
    <td class="tg-nrix">35.0 </td>
    <td class="tg-nrix">40.3 </td>
    <td class="tg-nrix">36.5 </td>
    <td class="tg-nrix">42.2 </td>
    <td class="tg-nrix">42.1 </td>
    <td class="tg-nrix">32.04</td>
    <td class="tg-nrix">27.7 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SRN</td>
    <td class="tg-wn3b">CVPR20</td>
    <td class="tg-wn3b">ResNet+FPN</td>
    <td class="tg-wn3b">94.8 </td>
    <td class="tg-wn3b">91.5 </td>
    <td class="tg-wn3b">95.5 </td>
    <td class="tg-wn3b">82.7 </td>
    <td class="tg-wn3b">85.1 </td>
    <td class="tg-wn3b">87.8</td>
    <td class="tg-wn3b">89.57</td>
    <td class="tg-wn3b">63.4 </td>
    <td class="tg-wn3b">25.3 </td>
    <td class="tg-wn3b">34.1 </td>
    <td class="tg-wn3b">28.7 </td>
    <td class="tg-wn3b">56.5 </td>
    <td class="tg-wn3b">26.7 </td>
    <td class="tg-wn3b">46.3 </td>
    <td class="tg-wn3b">40.14</td>
    <td class="tg-wn3b">54.7 </td>
  </tr>
  <tr>
    <td class="tg-nrix">SEED*</td>
    <td class="tg-nrix">CVPR20</td>
    <td class="tg-nrix">ResNet+LSTM</td>
    <td class="tg-nrix">93.8 </td>
    <td class="tg-nrix">89.6 </td>
    <td class="tg-nrix">92.8 </td>
    <td class="tg-nrix">80.0 </td>
    <td class="tg-nrix">81.4 </td>
    <td class="tg-nrix">83.6</td>
    <td class="tg-nrix">86.87</td>
    <td class="tg-nrix">40.4 </td>
    <td class="tg-nrix">15.5 </td>
    <td class="tg-nrix">32.1 </td>
    <td class="tg-nrix">32.5 </td>
    <td class="tg-nrix">54.8 </td>
    <td class="tg-nrix">35.6 </td>
    <td class="tg-nrix">39.0 </td>
    <td class="tg-nrix">35.70 </td>
    <td class="tg-nrix">24.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">AutoSTR*</td>
    <td class="tg-wn3b">ECCV20</td>
    <td class="tg-wn3b">NAS+LSTM</td>
    <td class="tg-wn3b">94.7 </td>
    <td class="tg-wn3b">90.9 </td>
    <td class="tg-wn3b">94.2 </td>
    <td class="tg-wn3b">81.8 </td>
    <td class="tg-wn3b">81.7 </td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">47.7 </td>
    <td class="tg-wn3b">17.9 </td>
    <td class="tg-wn3b">30.8 </td>
    <td class="tg-wn3b">36.2 </td>
    <td class="tg-wn3b">64.2 </td>
    <td class="tg-wn3b">38.7 </td>
    <td class="tg-wn3b">41.3 </td>
    <td class="tg-wn3b">39.54</td>
    <td class="tg-wn3b">6.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">RoScanner</td>
    <td class="tg-nrix">ECCV20</td>
    <td class="tg-nrix">ResNet</td>
    <td class="tg-nrix">95.3 </td>
    <td class="tg-nrix">88.1 </td>
    <td class="tg-nrix">94.8 </td>
    <td class="tg-nrix">77.1 </td>
    <td class="tg-nrix">79.5 </td>
    <td class="tg-nrix">90.3</td>
    <td class="tg-nrix">87.52</td>
    <td class="tg-nrix">43.6 </td>
    <td class="tg-nrix">7.9 </td>
    <td class="tg-nrix">41.2 </td>
    <td class="tg-nrix">42.6 </td>
    <td class="tg-nrix">44.9 </td>
    <td class="tg-nrix">46.9 </td>
    <td class="tg-nrix">39.5 </td>
    <td class="tg-nrix">38.09</td>
    <td class="tg-nrix">48.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">ABINet</td>
    <td class="tg-wn3b">CVPR21</td>
    <td class="tg-wn3b">ResNet+TF3</td>
    <td class="tg-wn3b">96.2 </td>
    <td class="tg-wn3b">93.5 </td>
    <td class="tg-wn3b">97.4 </td>
    <td class="tg-wn3b">86.0 </td>
    <td class="tg-wn3b">89.3 </td>
    <td class="tg-wn3b">89.2</td>
    <td class="tg-wn3b">91.93</td>
    <td class="tg-wn3b">59.5 </td>
    <td class="tg-wn3b">12.7 </td>
    <td class="tg-wn3b">43.3 </td>
    <td class="tg-wn3b">38.3 </td>
    <td class="tg-wn3b">62.0 </td>
    <td class="tg-wn3b">50.8 </td>
    <td class="tg-wn3b">55.6 </td>
    <td class="tg-wn3b">46.03</td>
    <td class="tg-wn3b">36.7 </td>
  </tr>
  <tr>
    <td class="tg-nrix">VisionLAN</td>
    <td class="tg-nrix">ICCV21</td>
    <td class="tg-nrix">ResNet+TF3</td>
    <td class="tg-nrix">95.8 </td>
    <td class="tg-nrix">91.7 </td>
    <td class="tg-nrix">95.7 </td>
    <td class="tg-nrix">83.7 </td>
    <td class="tg-nrix">86.0 </td>
    <td class="tg-nrix">88.5</td>
    <td class="tg-nrix">90.23</td>
    <td class="tg-nrix">57.7 </td>
    <td class="tg-nrix">14.2 </td>
    <td class="tg-nrix">47.8 </td>
    <td class="tg-nrix">48.0 </td>
    <td class="tg-nrix">64.0 </td>
    <td class="tg-nrix">47.9 </td>
    <td class="tg-nrix">52.1 </td>
    <td class="tg-nrix">47.39</td>
    <td class="tg-nrix">32.8 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">PARSeq*</td>
    <td class="tg-wn3b">ECCV22</td>
    <td class="tg-wn3b">ViT-S</td>
    <td class="tg-wn3b">97.0 </td>
    <td class="tg-wn3b">93.6 </td>
    <td class="tg-wn3b">97.0 </td>
    <td class="tg-wn3b">86.5 </td>
    <td class="tg-wn3b">88.9 </td>
    <td class="tg-wn3b">92.2</td>
    <td class="tg-wn3b">92.53</td>
    <td class="tg-wn3b">63.9 </td>
    <td class="tg-wn3b">16.7 </td>
    <td class="tg-wn3b">52.5 </td>
    <td class="tg-wn3b">54.3 </td>
    <td class="tg-wn3b">68.2 </td>
    <td class="tg-wn3b">55.9 </td>
    <td class="tg-wn3b">56.9 </td>
    <td class="tg-wn3b">52.62</td>
    <td class="tg-wn3b">23.8 </td>
  </tr>
  <tr>
    <td class="tg-nrix">MATRN</td>
    <td class="tg-nrix">ECCV22</td>
    <td class="tg-nrix">ResNet+TF3</td>
    <td class="tg-nrix">96.6 </td>
    <td class="tg-nrix">95.0 </td>
    <td class="tg-nrix">97.9 </td>
    <td class="tg-nrix">86.6 </td>
    <td class="tg-nrix">90.6 </td>
    <td class="tg-nrix">93.5</td>
    <td class="tg-nrix">93.37</td>
    <td class="tg-nrix">63.1 </td>
    <td class="tg-nrix">13.4 </td>
    <td class="tg-nrix">43.8 </td>
    <td class="tg-nrix">41.9 </td>
    <td class="tg-nrix">66.4 </td>
    <td class="tg-nrix">53.2 </td>
    <td class="tg-nrix">57.0 </td>
    <td class="tg-nrix">48.40 </td>
    <td class="tg-nrix">44.2 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">MGP-STR*</td>
    <td class="tg-wn3b">ECCV22</td>
    <td class="tg-wn3b">ViT-B</td>
    <td class="tg-wn3b">96.4 </td>
    <td class="tg-wn3b">94.7 </td>
    <td class="tg-wn3b">97.3 </td>
    <td class="tg-wn3b">87.2 </td>
    <td class="tg-wn3b">91.0 </td>
    <td class="tg-wn3b">90.3</td>
    <td class="tg-wn3b">92.82</td>
    <td class="tg-wn3b">55.2 </td>
    <td class="tg-wn3b">14.0 </td>
    <td class="tg-wn3b">52.8 </td>
    <td class="tg-wn3b">48.5 </td>
    <td class="tg-wn3b">65.2 </td>
    <td class="tg-wn3b">48.8 </td>
    <td class="tg-wn3b">59.1 </td>
    <td class="tg-wn3b">49.09</td>
    <td class="tg-wn3b">148.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">LevOCR*</td>
    <td class="tg-nrix">ECCV22</td>
    <td class="tg-nrix">ResNet+TF3</td>
    <td class="tg-nrix">96.6 </td>
    <td class="tg-nrix">94.4 </td>
    <td class="tg-nrix">96.7 </td>
    <td class="tg-nrix">86.5 </td>
    <td class="tg-nrix">88.8 </td>
    <td class="tg-nrix">90.6</td>
    <td class="tg-nrix">92.27</td>
    <td class="tg-nrix">52.8 </td>
    <td class="tg-nrix">10.7 </td>
    <td class="tg-nrix">44.8 </td>
    <td class="tg-nrix">51.9 </td>
    <td class="tg-nrix">61.3 </td>
    <td class="tg-nrix">54.0 </td>
    <td class="tg-nrix">58.1 </td>
    <td class="tg-nrix">47.66</td>
    <td class="tg-nrix">109.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">CornerTF*</td>
    <td class="tg-wn3b">ECCV22</td>
    <td class="tg-wn3b">CornerEncoder</td>
    <td class="tg-wn3b">95.9 </td>
    <td class="tg-wn3b">94.6 </td>
    <td class="tg-wn3b">97.8 </td>
    <td class="tg-wn3b">86.5 </td>
    <td class="tg-wn3b">91.5 </td>
    <td class="tg-wn3b">92.0 </td>
    <td class="tg-wn3b">93.05</td>
    <td class="tg-wn3b">62.9 </td>
    <td class="tg-wn3b">18.6 </td>
    <td class="tg-wn3b">56.1 </td>
    <td class="tg-wn3b">58.5 </td>
    <td class="tg-wn3b">68.6 </td>
    <td class="tg-wn3b">59.7 </td>
    <td class="tg-wn3b">61.0 </td>
    <td class="tg-wn3b">55.07</td>
    <td class="tg-wn3b">86.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">CPPD</td>
    <td class="tg-nrix">Preprint</td>
    <td class="tg-nrix">SVTR-B</td>
    <td class="tg-nrix">97.6 </td>
    <td class="tg-nrix">95.5 </td>
    <td class="tg-nrix">98.2 </td>
    <td class="tg-nrix">87.9 </td>
    <td class="tg-nrix">90.9 </td>
    <td class="tg-nrix">92.7</td>
    <td class="tg-nrix">93.80 </td>
    <td class="tg-nrix">65.5 </td>
    <td class="tg-nrix">18.6 </td>
    <td class="tg-nrix">56.0 </td>
    <td class="tg-nrix">61.9 </td>
    <td class="tg-nrix">71.0 </td>
    <td class="tg-nrix">57.5 </td>
    <td class="tg-nrix">65.8 </td>
    <td class="tg-nrix">56.63</td>
    <td class="tg-nrix">26.8 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SIGA*</td>
    <td class="tg-wn3b">CVPR23</td>
    <td class="tg-wn3b">ViT-B</td>
    <td class="tg-wn3b">96.6 </td>
    <td class="tg-wn3b">95.1 </td>
    <td class="tg-wn3b">97.8 </td>
    <td class="tg-wn3b">86.6 </td>
    <td class="tg-wn3b">90.5 </td>
    <td class="tg-wn3b">93.1</td>
    <td class="tg-wn3b">93.28</td>
    <td class="tg-wn3b">59.9 </td>
    <td class="tg-wn3b">22.3 </td>
    <td class="tg-wn3b">49.0 </td>
    <td class="tg-wn3b">50.8 </td>
    <td class="tg-wn3b">66.4 </td>
    <td class="tg-wn3b">58.4 </td>
    <td class="tg-wn3b">56.2 </td>
    <td class="tg-wn3b">51.85</td>
    <td class="tg-wn3b">113.0 </td>
  </tr>
  <tr>
    <td class="tg-nrix">CCD*</td>
    <td class="tg-nrix">ICCV23</td>
    <td class="tg-nrix">ViT-B</td>
    <td class="tg-nrix">97.2 </td>
    <td class="tg-nrix">94.4 </td>
    <td class="tg-nrix">97.0 </td>
    <td class="tg-nrix">87.6 </td>
    <td class="tg-nrix">91.8 </td>
    <td class="tg-nrix">93.3</td>
    <td class="tg-nrix">93.55</td>
    <td class="tg-nrix">66.6 </td>
    <td class="tg-nrix">24.2 </td>
    <td class="tg-10jc">63.9 </td>
    <td class="tg-nrix">64.8 </td>
    <td class="tg-nrix">74.8 </td>
    <td class="tg-nrix">62.4 </td>
    <td class="tg-nrix">64.0 </td>
    <td class="tg-nrix">60.10 </td>
    <td class="tg-nrix">52.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">LISTER*</td>
    <td class="tg-wn3b">ICCV23</td>
    <td class="tg-wn3b">FocalNet-B</td>
    <td class="tg-wn3b">96.9 </td>
    <td class="tg-wn3b">93.8 </td>
    <td class="tg-wn3b">97.9 </td>
    <td class="tg-wn3b">87.5 </td>
    <td class="tg-wn3b">89.6 </td>
    <td class="tg-wn3b">90.6</td>
    <td class="tg-wn3b">92.72</td>
    <td class="tg-wn3b">56.5 </td>
    <td class="tg-wn3b">17.2 </td>
    <td class="tg-wn3b">52.8 </td>
    <td class="tg-wn3b">63.5 </td>
    <td class="tg-wn3b">63.2 </td>
    <td class="tg-wn3b">59.6 </td>
    <td class="tg-wn3b">65.4 </td>
    <td class="tg-wn3b">54.05</td>
    <td class="tg-wn3b">49.9 </td>
  </tr>
  <tr>
    <td class="tg-nrix">LPV-B*</td>
    <td class="tg-nrix">IJCAI23</td>
    <td class="tg-nrix">SVTR-B</td>
    <td class="tg-nrix">97.3 </td>
    <td class="tg-nrix">94.6 </td>
    <td class="tg-nrix">97.6 </td>
    <td class="tg-nrix">87.5 </td>
    <td class="tg-nrix">90.9 </td>
    <td class="tg-nrix">94.8</td>
    <td class="tg-nrix">93.78</td>
    <td class="tg-nrix">68.3 </td>
    <td class="tg-nrix">21.0 </td>
    <td class="tg-nrix">59.6 </td>
    <td class="tg-nrix">65.1 </td>
    <td class="tg-nrix">76.2 </td>
    <td class="tg-nrix">63.6 </td>
    <td class="tg-nrix">62.0 </td>
    <td class="tg-nrix">59.40 </td>
    <td class="tg-nrix">35.1 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">CDistNet*</td>
    <td class="tg-wn3b">IJCV24</td>
    <td class="tg-wn3b">ResNet+TF3</td>
    <td class="tg-wn3b">96.4 </td>
    <td class="tg-wn3b">93.5 </td>
    <td class="tg-wn3b">97.4 </td>
    <td class="tg-wn3b">86.0 </td>
    <td class="tg-wn3b">88.7 </td>
    <td class="tg-wn3b">93.4</td>
    <td class="tg-wn3b">92.57</td>
    <td class="tg-wn3b">69.3 </td>
    <td class="tg-wn3b">24.4 </td>
    <td class="tg-wn3b">49.8 </td>
    <td class="tg-wn3b">55.6 </td>
    <td class="tg-wn3b">72.8 </td>
    <td class="tg-wn3b">64.3 </td>
    <td class="tg-wn3b">58.5 </td>
    <td class="tg-wn3b">56.38</td>
    <td class="tg-wn3b">65.5 </td>
  </tr>
  <tr>
    <td class="tg-nrix">CAM*</td>
    <td class="tg-nrix">PR24</td>
    <td class="tg-nrix">ConvNeXtV2-B</td>
    <td class="tg-nrix">97.4 </td>
    <td class="tg-10jc">96.1 </td>
    <td class="tg-nrix">97.2 </td>
    <td class="tg-nrix">87.8 </td>
    <td class="tg-nrix">90.6 </td>
    <td class="tg-nrix">92.4</td>
    <td class="tg-nrix">93.58</td>
    <td class="tg-nrix">63.1 </td>
    <td class="tg-nrix">19.4 </td>
    <td class="tg-nrix">55.4 </td>
    <td class="tg-nrix">58.5 </td>
    <td class="tg-nrix">72.7 </td>
    <td class="tg-nrix">51.4 </td>
    <td class="tg-nrix">57.4 </td>
    <td class="tg-nrix">53.99</td>
    <td class="tg-nrix">135.0 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">BUSNet</td>
    <td class="tg-wn3b">AAAI24</td>
    <td class="tg-wn3b">ViT-S</td>
    <td class="tg-wn3b">96.2 </td>
    <td class="tg-wn3b">95.5 </td>
    <td class="tg-s1e0">98.3 </td>
    <td class="tg-wn3b">87.2 </td>
    <td class="tg-s1e0">91.8 </td>
    <td class="tg-wn3b">91.3</td>
    <td class="tg-wn3b">93.38</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">56.8 </td>
  </tr>
  <tr>
    <td class="tg-nrix">DCTC</td>
    <td class="tg-nrix">AAAI24</td>
    <td class="tg-nrix">SVTR-L</td>
    <td class="tg-nrix">96.9 </td>
    <td class="tg-nrix">93.7 </td>
    <td class="tg-nrix">97.4 </td>
    <td class="tg-nrix">87.3 </td>
    <td class="tg-nrix">88.5 </td>
    <td class="tg-nrix">92.3</td>
    <td class="tg-nrix">92.68</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">40.8 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">OTE</td>
    <td class="tg-wn3b">CVPR24</td>
    <td class="tg-wn3b">SVTR-B</td>
    <td class="tg-wn3b">96.4 </td>
    <td class="tg-wn3b">95.5 </td>
    <td class="tg-wn3b">97.4 </td>
    <td class="tg-wn3b">87.2 </td>
    <td class="tg-wn3b">89.6 </td>
    <td class="tg-wn3b">92.4</td>
    <td class="tg-wn3b">93.08</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">-</td>
    <td class="tg-wn3b">25.2 </td>
  </tr>
  <tr>
    <td class="tg-nrix">CFF</td>
    <td class="tg-nrix">IJCAI24</td>
    <td class="tg-nrix">CEFE</td>
    <td class="tg-nrix">97.6 </td>
    <td class="tg-nrix">94.3 </td>
    <td class="tg-nrix">97.9 </td>
    <td class="tg-nrix">86.9 </td>
    <td class="tg-10jc">91.8 </td>
    <td class="tg-nrix">95.5</td>
    <td class="tg-nrix">94.00 </td>
    <td class="tg-nrix">70.0 </td>
    <td class="tg-nrix">20.8</td>
    <td class="tg-nrix">62.4</td>
    <td class="tg-10jc">72.0 </td>
    <td class="tg-nrix">75.2</td>
    <td class="tg-nrix">65.7</td>
    <td class="tg-nrix">65.1</td>
    <td class="tg-nrix">61.60 </td>
    <td class="tg-nrix">23.9 </td>
  </tr>
  <tr>
    <td class="tg-wn3b" rowspan="3">CTCs</td>
    <td class="tg-wn3b">CRNN</td>
    <td class="tg-wn3b">TPAMI16</td>
    <td class="tg-wn3b">ResNet+LSTM</td>
    <td class="tg-wn3b">82.9 </td>
    <td class="tg-wn3b">81.6 </td>
    <td class="tg-wn3b">91.1 </td>
    <td class="tg-wn3b">69.4 </td>
    <td class="tg-wn3b">70.0 </td>
    <td class="tg-wn3b">65.5</td>
    <td class="tg-wn3b">76.75</td>
    <td class="tg-wn3b">7.5</td>
    <td class="tg-wn3b">0.9</td>
    <td class="tg-wn3b">20.7</td>
    <td class="tg-wn3b">25.6</td>
    <td class="tg-wn3b">13.9</td>
    <td class="tg-wn3b">25.6</td>
    <td class="tg-wn3b">32.0 </td>
    <td class="tg-wn3b">18.03</td>
    <td class="tg-wn3b">8.3 </td>
  </tr>
  <tr>
    <td class="tg-nrix">SVTR*</td>
    <td class="tg-nrix">IJCAI22</td>
    <td class="tg-nrix">SVTR-B</td>
    <td class="tg-nrix">96.0 </td>
    <td class="tg-nrix">91.5 </td>
    <td class="tg-nrix">97.1 </td>
    <td class="tg-nrix">85.2 </td>
    <td class="tg-nrix">89.9 </td>
    <td class="tg-nrix">91.7</td>
    <td class="tg-nrix">91.90 </td>
    <td class="tg-nrix">69.8</td>
    <td class="tg-10jc">37.7</td>
    <td class="tg-nrix">47.9</td>
    <td class="tg-nrix">61.4</td>
    <td class="tg-nrix">66.8</td>
    <td class="tg-nrix">44.8</td>
    <td class="tg-nrix">61.0 </td>
    <td class="tg-nrix">55.63</td>
    <td class="tg-nrix">24.6 </td>
  </tr>
  <tr>
    <td class="tg-wn3b">SVTRv2</td>
    <td class="tg-wn3b">Preprint</td>
    <td class="tg-wn3b">SVTRv2-B</td>
    <td class="tg-s1e0">97.7 </td>
    <td class="tg-wn3b">94.0 </td>
    <td class="tg-wn3b">97.3 </td>
    <td class="tg-s1e0">88.1 </td>
    <td class="tg-wn3b">91.2 </td>
    <td class="tg-s1e0">95.8</td>
    <td class="tg-s1e0">94.02</td>
    <td class="tg-s1e0">74.6</td>
    <td class="tg-wn3b">25.2</td>
    <td class="tg-wn3b">57.6</td>
    <td class="tg-wn3b">69.7</td>
    <td class="tg-sjbb">77.9</td>
    <td class="tg-s1e0">68.0 </td>
    <td class="tg-s1e0">66.9</td>
    <td class="tg-s1e0">62.83</td>
    <td class="tg-wn3b">19.8 </td>
  </tr>
</tbody></table>

**Note**: * represents that the results on Union14M-Benchmarks are evaluated using the model they released.

## Citation

```bibtex
@article{Du2024SVTRv4,
  title     = {SVTRv2: Scene Text Recognition with a Single Visual Model},
  author    = {Yongkun Du, Zhineng Chen\*, Hongtao Xie, Caiyan Jia, Yu-Gang Jiang},
  year      = {2024}
}
```

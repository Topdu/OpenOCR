# CMER: Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline

[Paper](https://arxiv.org/abs/2512.13731)

## Introduction

**CMER (Complex Mathematical Expression Recognition)** is a comprehensive project designed to address the challenges in recognizing complex, multi-line, and high-density mathematical formulas. While existing methods perform well on simple expressions, they often struggle with complicated spatial layouts and long token sequences.

This repository hosts the official implementation of **CMERNet**, a specialized model built upon an encoder-decoder architecture. It introduces a novel expression tokenizer and **Structured Mathematical Language** representation to explicitly model hierarchical and spatial structures.

## Get Started with CMER

### Dependencies

To run CMER, ensure your environment meets the following requirements:

- **Python**: 3.10+
- **PyTorch**: 2.5.1+ (CUDA 12.x recommended)
- **Core Libraries**: `transformers`, `webdataset`, `albumentations`, `evaluate`, `levenshtein`
- **requirements**: ./configs/rec/cmer/requirements.txt

## Data Download

We provide both the benchmark dataset for evaluation and the large-scale dataset used for training.

### CMER-Bench (Evaluation)

**CMER-Bench** is a carefully constructed benchmark that categorizes expressions into three difficulty levels: easy, moderate, and complex. It is designed to evaluate the robustness of MER models on complicated spatial layouts.

- **Download Link**: [Hugging Face - CMER-Bench1.5](https://huggingface.co/datasets/Baitlo/CMER-Bench1.5)

### CMER-3M (Training)

**CMER-3M** is a large-scale dataset emphasizing the recognition of complex mathematical expressions, providing rich and diverse samples to support the development of accurate MER models.

- **Download Link**: [Hugging Face - CMER-3M](https://huggingface.co/datasets/Baitlo/CMER-3M) *(Note: This dataset is currently not open-sourced and will be available soon.)*

*After downloading, please extract the datasets and update the `data_dir` paths in your configuration file accordingly.*

## Configuration

The model training and inference are controlled by a YAML configuration file (e.g., `./configs/rec/cmer/cmer.yml`). Before running the code, please modify the relevant parameters to fit your environment:

- **Global Settings**:
  - `device`: Computing device (e.g., `gpu`).
  - `epoch_num`: Total training epochs.
  - `output_dir`: Directory to save checkpoints and logs.
  - `pretrained_model`: Path to the trained model weights (e.g., `.safetensors` or `.pth`) for fine-tuning or inference.
  - `infer_img`: Directory containing images to be recognized during inference.
  - `save_res_path`: Path to save the recognition results file.
- **Data Paths**:
  - `Train.dataset.data_dir`: Path to training images.
  - `Train.dataset.label_file_list`: Path to training labels.
    - Label File Format: The label file should be a .txt file where each line contains the image filename and the LaTeX formula, separated by a tab (\\t).
    - Example: 00000000\_\_35270b0175.png \\begin{equation}{\\cal D}=\\frac{N}{p\\bar{\\ell}_{s}^{2}\\rho_{0}}:\\frac{\\partial}{\\partial t}-\\frac{\\partial^{2}}{\\partial j^{2}}\\end{equation}
  - `Eval.dataset.data_dir`: Path to validation images.
  - `Eval.dataset.label_file_list`: Path to validation labels.
- **Model Architecture**:
  - Defined under `Architecture`, including `vision_config` (encoder settings) and `decoder_config` (decoder settings).

## Train

CMER supports distributed training using `torch.distributed.launch`.

1. **Prepare Data**: Ensure your training data and label files are correctly set in `configs/rec/cmer/cmer.yml` under the `Train` section.

2. **Start Training**:
   Run the following command to start distributed training (example uses 4 GPUs):

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c ./configs/rec/cmer/cmer.yml
   ```

## Inference

You can evaluate the model or perform inference on custom images using the `tools/infer_cmer.py` script.

1. **Configure Inference**:
   Modify `configs/rec/cmer/cmer.yml` to point to your target images and model weights:

   - `Global.pretrained_model`: Path to the trained model weight file.
   - `Global.infer_img`: Directory containing images to be recognized.
   - `Global.save_res_path`: Path to save the recognition results.

2. **Run Inference**:
   Execute the following command:

   ```bash
   python ./tools/infer_cmer.py -c ./configs/rec/cmer/cmer.yml
   ```

```
```

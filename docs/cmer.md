# CMER: Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline

[Paper](https://arxiv.org/abs/2512.13731)


## Introduction

**CMER (Complex Mathematical Expression Recognition)** is a comprehensive project designed to address the challenges in recognizing complex, multi-line, and high-density mathematical formulas. While existing methods perform well on simple expressions, they often struggle with complicated spatial layouts and long token sequences.

This repository hosts the official implementation of **CMERNet**, a specialized model built upon an encoder-decoder architecture. It introduces a novel expression tokenizer and **Structured Mathematical Language** representation to explicitly model hierarchical and spatial structures.

## Get Started with CMER

### Dependencies

To run CMER, ensure your environment meets the following requirements:

*   **Python**: 3.10+
*   **PyTorch**: 2.5.1+ (CUDA 12.x recommended)
*   **Core Libraries**: `transformers`, `webdataset`, `albumentations`, `evaluate`, `levenshtein`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Baitlo/CMER.git
    cd CMER
    ```

2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The model training and inference are controlled by a YAML configuration file (e.g., `./configs/rec/cmer/cmer.yml`). Before running the code, please modify the relevant parameters to fit your environment:

*   **Global Settings**:
    *   `device`: Computing device (e.g., `gpu`).
    *   `epoch_num`: Total training epochs.
    *   `output_dir`: Directory to save checkpoints and logs.
    *   `infer_weight`: Path to the trained model weights for inference.
*   **Data Paths**:
    *   `Train.dataset.data_dir`: Path to training images.
    *   `Train.dataset.label_file_list`: Path to training labels.
    *   `Eval.dataset.data_dir`: Path to validation images.
    *   `Eval.dataset.label_file_list`: Path to validation labels.
*   **Model Architecture**:
    *   Defined under `Architecture`, including `vision_config` and `decoder_config`.

## Train

CMER supports distributed training using `torch.distributed.launch`.

1.  **Prepare Data**: Ensure your training data and label files are correctly set in `configs/rec/cmer/cmer.yml` under the `Train` section.

2.  **Start Training**:
    Run the following command to start distributed training (example uses 4 GPUs):

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c ./configs/rec/cmer/cmer.yml --task formula_rec
    ```

## Inference

You can evaluate the model or perform inference on custom images using the `tools/infer_cmer.py` script.

1.  **Configure Inference**:
    Modify `configs/rec/cmer/cmer.yml` to point to your target images and model weights:
    *   `Global.infer_weight`: Path to the `.pth` or `.safetensors` model file.
    *   `Global.infer_img_root`: Directory containing images to be recognized.
    *   `Global.infer_label_file`: (Optional) Label file for evaluation.
    *   `Global.save_res_path`: Path to save the recognition results.

2.  **Run Inference**:
    Execute the following command:
    ```bash
    python ./tools/infer_cmer.py -c ./configs/rec/cmer/cmer.yml
    ```

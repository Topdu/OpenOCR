# UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters

\[[Paper](https://arxiv.org/pdf/2512.21095)\] \[[ModelScope Model](https://www.modelscope.cn/models/topdktu/unirec-0.1b)\] \[[HuggingFace Model](https://huggingface.co/topdu/unirec-0.1b)\] \[[ModelScope Demo](https://www.modelscope.cn/studios/topdktu/OpenOCR-UniRec-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-UniRec-Demo)\] \[[Local Demo](#local-demo)\] \[[UniRec40M Dataset](https://huggingface.co/datasets/topdu/UniRec40M)\]

## Introduction

**UniRec-0.1B** is a unified recognition model with only 0.1B parameters, designed for high-accuracy and efficient recognition of plain text (words, lines, paragraphs), mathematical formulas (single-line, multi-line), and mixed content in both Chinese and English. 

It addresses structural variability and semantic entanglement by using a hierarchical supervision training strategy and a semantic-decoupled tokenizer. Despite its small size, it achieves performance comparable to or better than much larger vision-language models.


## Get Started with the UniRec

### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocr python==3.10
conda activate openocr
# install gpu version torch >=1.13.0
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# or cpu version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
git clone https://github.com/Topdu/OpenOCR.git
```

### Downloding the UniRec Model from ModelScope or Hugging Face

```shell
cd OpenOCR
pip install -r requirements.txt
# download model from modelscope
modelscope download topdktu/unirec-0.1b --local_dir ./unirec-0.1b
# or download model from huggingface
huggingface-cli download topdu/unirec-0.1b --local-dir ./unirec-0.1b
```

### Inference

```shell
python tools/infer_rec.py --c ./configs/rec/unirec/focalsvtr_ardecoder_unirec.yml --o Global.infer_img=/path/img_fold or /path/img_file
```

### Local Demo

```shell
pip install gradio==4.20.0
python demo_unirec.py
```

### Training

Additional dependencies:

```shell
pip install PyMuPDF
pip install pdf2image
pip install numpy==1.26.4
pip install albumentations==1.4.24
pip install transformers==4.49.0
pip install -U flash-attn --no-build-isolation
```

It is recommended to organize your working directory as follows:

```shell
|-UniRec40M    # Main directory for UniRec40M dataset
|-OpenOCR      # Directory for OpenOCR-related files
|-evaluation   # Directory for evaluation dataset
```

Download the UniRec40M dataset from Hugging Face

```shell
# downloading small data for quickly training
huggingface-cli download topdu/UniRec40M --include "hiertext_lmdb/**" --repo-type dataset --local-dir ./UniRec40M/
huggingface-cli download topdu/OpenOCR-Data --include "evaluation/**" --repo-type dataset --local-dir ./
```

Run the following command to train the model quickly:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=23333 --nproc_per_node=8 tools/train_rec.py --c configs/rec/unirec/focalsvtr_ardecoder_unirec.yml
```

Downloading the full dataset requires 3.5 TB of available storage space. Then, you need to merge the split files named `data.mdb.part_*` (located in `HWDB2Train`, `ch_pdf_lmdb`, and `en_pdf_lmdb`) into a single `data.mdb` file. Execute the commands below step by step:

```shell
# downloading full data
huggingface-cli download topdu/UniRec40M --repo-type dataset --local-dir ./UniRec40M/
cd UniRec40M/HWDB2Train/image_lmdb & cat data.mdb.part_* > data.mdb
cd UniRec40M/ch_pdf_lmdb & cat data.mdb.part_* > data.mdb
cd UniRec40M/en_pdf_lmdb & cat data.mdb.part_* > data.mdb
```

And modify the `configs/rec/unirec/focalsvtr_ardecoder_unirec.yml` file as follows:

```yaml
...
Train:
  dataset:
    name: NaSizeDataSet
    divided_factor: &divided_factor [64, 64] # w, h
    max_side: &max_side [960, 1408] # [64*30, 64*44] # w, h [960, 1408] #
    root_path: path/to/UniRec40M
    add_return: True
    zoom_min_factor: 4
    use_zoom: True
    all_data: True
    test_data: False
    use_aug: True
    use_linedata: True
    transforms:
      - UniRecLabelEncode: # Class handling label
          max_text_length: *max_text_length
          vlmocr: True
          tokenizer_path: *vlm_ocr_config # path to tokenizer, e.g. 'vocab.json', 'merges.txt'
      - KeepKeys:
          keep_keys: ['image', 'label', 'length'] # dataloader will return list in this order
  sampler:
    name: NaSizeSampler
    # divide_factor: to ensure the width and height dimensions can be devided by downsampling multiple
    min_bs: 1
    max_bs: 24
  loader:
    shuffle: True
    batch_size_per_card: 64
    drop_last: True
    num_workers: 8
...
```

## Citation

If you find our method useful for your research, please cite:

```bibtex
@article{du2025unirec,
  title={UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters},
  author={Yongkun Du and Zhineng Chen and Yazhen Xie and Weikang Bai and Hao Feng and Wei Shi and Yuchen Su and Can Huang and Yu-Gang Jiang},
  journal={arXiv preprint arXiv:2512.21095},
  year={2025}
}
```

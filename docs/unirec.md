# UniRec: Unified Text and Formula Recognition Across Granularities

\[Paper coming soon\] \[[Model](https://huggingface.co/topdu/unirec_100m)\] \[[ModelScope Demo](https://www.modelscope.cn/studios/topdktu/OpenOCR-UniRec-Demo)\] \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-UniRec-Demo)\] \[[Local Demo](#local-demo)\]

## Introduction

**UniRec** is good at recognizing plain text (words, lines, paragraphs), formulas (single-line, multi-line), and mixed text-and-formulas content. You only need to use a screenshot tool to select the text area from a paper and paste it into the \[[ModelScope Demo](https://www.modelscope.cn/studios/topdktu/OpenOCR-UniRec-Demo)\] or \[[Hugging Face Demo](https://huggingface.co/spaces/topdu/OpenOCR-UniRec-Demo)\]. After clicking Run, the recognition will be completed automatically

## Get Started with the UniRec

### Dependencies:

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
conda create -n openocr python==3.9
conda activate openocr
# install gpu version torch >=1.13.0
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# or cpu version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
git clone https://github.com/Topdu/OpenOCR.git
```

### Downloding the UniRec Model by Hugging Face

```shell

cd OpenOCR
pip install -r requirements.txt
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/topdu/unirec_100m
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

## Citation

If you find our method useful for your reserach, please cite:

```bibtex

```

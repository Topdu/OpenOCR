# OpenOCR(Paddle)

基于深度学习的场景文本检测与识别方法训练、评估、测试Benchmark

## 一期（STR）：

TODO
- [✅] SVTR训练、评估、测试
- [✅] CPPD训练，评估，测试
- [❌] CRNN
- [❌] ViT-CTC
- [❌] ResNet+En+CTC
- [❌] NRTR
- [❌] ABINet
- [❌] PARSeq
- [❌] STN
- [❌] SAR
- [❌] ASTER
- [❌] RobustScanner
- [❌] SRN
- [❌] VisionLAN
- [❌] LPV

### 训练准备

- 数据集
从PARSeq提供的数据集下载，链接为：https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE

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

总体流程为：

1、先Fork OpenOCR 项目到自己Github仓库中。

2、git clone -b develop https://github.com/自己的用户名/OpenOCR.git （注意每次git clone 之前要保证自己的仓库是最新代码）。

3、参考svtrnet_ctc和svtr_base_cppd的代码结构，将新算法的preprocess、modeling.encoder、modeling.decoder、optimizer、loss、postprocess添加到代码中。

4、安装pre-commit，执行代码风格检查。
```
pip install pre-commit
pre-commit install
```

5、将新添加的算法训练、评估、测试跑通后，按照github提交commit的流程向源仓库提交PR。

## 二期（STD）：


## 三期（E2E）：


## 提交代码


Thanks:

1. https://github.com/WenmuZhou/PytorchOCR
2. https://github.com/PaddlePaddle/PaddleOCR
3. https://github.com/frotms/PaddleOCR2Pytorch

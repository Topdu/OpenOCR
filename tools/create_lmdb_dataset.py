import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import io
from PIL import Image
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """


def get_datalist(data_dir, data_path, max_len):
    """
    获取训练和验证的数据list
    :param data_dir: 数据集根目录
    :param data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    if isinstance(data_path, list):
        for p in data_path:
            train_data.extend(get_datalist(data_dir, p, max_len))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(),
                             desc=f'load data from {data_path}'):
                line = (line.strip('\n').replace('.jpg ', '.jpg\t').replace(
                    '.png ', '.png\t').split('\t'))
                if len(line) > 1:
                    img_path = os.path.join(data_dir, line[0].strip(' '))
                    label = line[1]
                    if len(label) > max_len:
                        continue
                    if os.path.exists(
                            img_path) and os.path.getsize(img_path) > 0:
                        train_data.append([str(img_path), label])
    return train_data


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list,
                                 desc=f'make dataset, save to {outputPath}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            buf = io.BytesIO(imageBin)
            w, h = Image.open(buf).size
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        whKey = 'wh-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[whKey] = (str(w) + '_' + str(h)).encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    data_dir = './Union14M-L/'
    # downloading the filtered_label_list from https://drive.google.com/drive/folders/1x1LC8C_W-Frl3sGV9i9_i_OD-bqNdodJ?usp=drive_link
    label_file_list = [
        './Union14M-L/train_annos/filter_jsonl_mmocr0.x/filter_train_challenging.jsonl.txt',
        './Union14M-L/train_annos/filter_jsonl_mmocr0.x/filter_train_easy.jsonl.txt',
        './Union14M-L/train_annos/filter_jsonl_mmocr0.x/filter_train_hard.jsonl.txt',
        './Union14M-L/train_annos/filter_jsonl_mmocr0.x/filter_train_medium.jsonl.txt',
        './Union14M-L/train_annos/filter_jsonl_mmocr0.x/filter_train_normal.jsonl.txt'
    ]
    save_path_root = './Union14M-L-LMDB-Filtered/'

    for data_list in label_file_list:
        save_path = save_path_root + data_list.split('/')[-1].split(
            '.')[0] + '/'
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        train_data_list = get_datalist(data_dir, data_list, 800)

        createDataset(train_data_list, save_path)

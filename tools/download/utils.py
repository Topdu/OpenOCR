import sys
import urllib
import ssl
from tqdm import tqdm
import os
import json
import csv
import cv2
import xml.etree.ElementTree as ET
from torchvision.datasets.utils import extract_archive
import numpy as np

def get_dataset_info(cfg):
    download_urls, filenames, check_validity = cfg["download_links"], cfg["filenames"], cfg["check_validity"]
    urls, filename_paths, unpack_paths = [], [], []
    for u, f in zip(download_urls, filenames):
        urls.append(u)
        unpack_path = os.path.join(cfg["root"], cfg["dataset_name"], f.split(".")[0])
        unpack_paths.append(unpack_path)
        filename_paths.append(os.path.join(unpack_path, f))
    return urls, filename_paths, unpack_paths, check_validity

async def download_torrent(save_path, magnet_link, dataset_name):
    try: 
        from torrentp import TorrentDownloader
    except ImportError:
        raise ValueError(f"The {dataset_name} dataset requires torrentp to be installed.")
    torrent_file = TorrentDownloader(file_path=magnet_link, save_path=save_path)
    await torrent_file.start_download(download_speed=0, upload_speed=0)

def _iiit_preprocess_str(cfg):
    try: 
        from scipy.io import loadmat
    except ImportError:
        raise ValueError("The IIIT dataset requires scipy to be installed.")
    
    data_path = os.path.join(cfg["root"], cfg["dataset_name"], "data", "IIIT5K")
    train_set = loadmat(os.path.join(data_path, "trainCharBound.mat"))["trainCharBound"][0]
    test_set = loadmat(os.path.join(data_path, "testCharBound.mat"))["testCharBound"][0]
    
    data = []
    for path, label, _ in train_set:
        path, label = path[0], label[0]
        if len(label) >= cfg["max_len"]:
            continue
        data.append(
            [os.path.join(data_path, path), str(label)]
        )

    for path, label, _ in test_set:
        path, label = path[0], label[0]
        if len(label) >= cfg["max_len"]:
            continue
        data.append(
            [os.path.join(data_path, path), str(label)]
        )

    return data

def _icdar13_preprocess_str(cfg):
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])
    train_images_path = os.path.join(data_dir, "train_images_anns")
    train_images_anns_path = os.path.join(train_images_path, "gt.txt")
    
    data = []  
    with open(train_images_anns_path, "r") as f:
        for l in f:
            path, label = l.strip().split(", ")
            label = label[1:-1]
            if len(label) >= cfg["max_len"]:
                continue
            data.append(
                [os.path.join(train_images_path, path), str(label)]
            )

    test_images_anns_path = os.path.join(data_dir, "test_anns", "test_anns.txt")
    with open(test_images_anns_path, "r") as f:
        for l in f:
            path, label = l.strip().split(", ")
            label = label[1:-1]
            if len(label) >= cfg["max_len"]:
                continue
            data.append(
                [os.path.join(data_dir, "test", path), str(label)]
            )
    return data

def _icdar15_preprocess_str(cfg):
    try:
        # We have to use PIL instead of opencv because of metadata rotations
        from PIL import Image
    except ImportError:
        raise ValueError("The ICDAR15 dataset requires scipy to be installed.")

    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])
    train_imgs_path = os.path.join(data_dir, "train_images")
    train_imgs_files = sorted(
        os.listdir(train_imgs_path), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    train_anns_path = os.path.join(data_dir, "train_localization_transcription")
    train_anns_files = sorted(
        os.listdir(train_anns_path), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    data = []
    idx = 0
    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)
    for img_path, label_path in zip(train_imgs_files, train_anns_files):
        with open(os.path.join(train_anns_path, label_path), "r", encoding="utf-8-sig") as f:
            img = Image.open(os.path.join(train_imgs_path, img_path))
            csvreader = csv.reader(f)
            for row in csvreader:
                word = row[-1]
                if len(word) >= cfg["max_len"] or word == "###":
                    continue
                # LT, RT, RB, LB, WORD
                l, t, r, b = int(row[0]), int(row[1]), int(row[2]), int(row[5])
                curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
                img.crop((l, t, r, b)).save(curr_remapped_path)
                data.append([curr_remapped_path, word])
                idx += 1

    return data

def _svt_preprocess_str(cfg):
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"], "data")

    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)

    data = []
    idx = 0

    img_anns_dir = os.path.join(data_dir, "svt1")

    for s in ["train", "test"]:
        tree = ET.parse(os.path.join(img_anns_dir, f"{s}.xml"))
        root = tree.getroot()
        for elem in root:
            label_dict = {
                "paths": [],
                "bboxes": [],
                "labels": [],
            }
            for subelem in elem:
                if subelem.tag == "imageName":
                    curr_img_path = subelem.text
                if subelem.tag == "taggedRectangles":
                    for subsubelem in subelem:
                        label_text = next(iter(subsubelem)).text
                        if len(label_text) >= cfg["max_len"]:
                            continue
                        label_dict["paths"].append(os.path.join(img_anns_dir, curr_img_path))
                        label_dict["bboxes"].append(
                            [
                                int(subsubelem.get("x")), 
                                int(subsubelem.get("y")), 
                                int(subsubelem.get("width")), 
                                int(subsubelem.get("height")),
                            ]
                        )
                        label_dict["labels"].append(label_text)

            for path, bbox, label in zip(label_dict["paths"], label_dict["bboxes"], label_dict["labels"]):
                l, t, w, h = bbox
                r, b = l + w, t + h
                
                img = cv2.imread(path)
                h, w, _ = img.shape
                
                l, t = max(0, l), max(0, t)
                r, b = min(w, r), min(h, b)
                
                img = img[t:b, l:r]
                curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
                cv2.imwrite(curr_remapped_path, img)
                data.append([curr_remapped_path, label])
                idx += 1
    return data

def _svtp_preprocess_str(cfg):
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"], "data", "Case-Sensitive-Scene-Text-Recognition-Datasets-master", "svtp_test")
    
    imgs_paths = sorted(
        os.listdir(os.path.join(data_dir, "IMG")), key=lambda x: int(x.split(".")[0])
    )
    anns_paths = sorted(
        os.listdir(os.path.join(data_dir, "label")), key=lambda x: int(x.split(".")[0])
    )

    data = []
    for path, ann_path in zip(imgs_paths, anns_paths):
        with open(os.path.join(data_dir, "label", ann_path), "r") as f:
            label = f.read().strip()
        data.append([
            os.path.join(data_dir, "IMG", path), 
            label
        ])

    return data

def _cute_preprocess_str(cfg):
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])

    with open(os.path.join(data_dir, "anns", "anns.json"), "r") as f:
        labels = json.load(f)["data_list"]

    data = []
    for d in labels:
        for i in d["instances"]:
            data.append([
                os.path.join(data_dir, "data", "timage", d["img_path"].split("/")[-1]), 
                i["text"]
            ])
    return data

def _sroie19_preprocess_str(cfg):
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])

    imgs_paths = sorted(os.listdir(os.path.join(data_dir, "data", "fulltext_test(361p)")), key=lambda x: x.split(".")[0])
    anns_paths = sorted(os.listdir(os.path.join(data_dir, "labels")), key=lambda x: x.split(".")[0])

    data = []
    idx = 0
    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)

    for ann_path, img_path in zip(anns_paths, imgs_paths):
        full_ann_path = os.path.join(data_dir, "labels", ann_path)
        img = cv2.imread(os.path.join(data_dir, "data", "fulltext_test(361p)", img_path))
        if ann_path == "X51006619503.txt":
            encoding = "iso-8859-1"
        else:
            encoding = "utf-8"
        with open(full_ann_path, "r", encoding=encoding) as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                if row == []:
                    continue
                word = row[-1]
                if len(word) >= cfg["max_len"]:
                    continue
                # LT, RT, RB, LB, WORD
                l, t, r, b = int(row[0]), int(row[1]), int(row[2]), int(row[5])
                if l  == r or t == b:
                    continue
                curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
                cv2.imwrite(curr_remapped_path, img[t:b, l:r])
                data.append([curr_remapped_path, word])
                idx += 1
    return data

def _textocr_preprocess_str(cfg):
    try:
        # Use PIL instead of opencv to mitigate risk of metadata rotation
        from PIL import Image
    except ImportError:
        raise ValueError("The ICDAR15 dataset requires scipy to be installed.")

    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])

    train_json = json.load(open(os.path.join(data_dir, "train_anns", "train_anns.json"), "r"))
    val_json = json.load(open(os.path.join(data_dir, "val_anns", "val_anns.json"), "r"))

    anns_train = {v["id"]:v for k, v in train_json["anns"].items()}
    anns_val = {v["id"]:v for k, v in val_json["anns"].items()}

    ann2img_train = {}
    for k, v in train_json["imgToAnns"].items():
        ann2img_train.update({vi:k for vi in v})

    ann2img_val = {}
    for k, v in val_json["imgToAnns"].items():
        ann2img_val.update({vi:k for vi in v})
    
    data = []
    idx = 0
    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)

    anns_jsons = [
        (anns_train, ann2img_train),
        (anns_val, ann2img_val),
    ]
    for ann_json, mapping_json in anns_jsons:
        for k, v in tqdm(ann_json.items()):
            corresponding_img = mapping_json[k]
            img_path = os.path.join(data_dir, "train_images", "train_images", f"{corresponding_img}.jpg")
            img = Image.open(img_path)
            bbox, word = v["bbox"], v["utf8_string"]
            # LT, RT, RB, LB
            l, t, w, h = [int(x) for x in bbox]
            r, b = l + w, t + h
            curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
            img.crop((l, t, r, b)).save(curr_remapped_path)
            data.append([curr_remapped_path, word])
            idx += 1
    return data

def _totaltext_preprocess_str(cfg):
    try: 
        from scipy.io import loadmat
        from PIL import Image
    except ImportError:
        raise ValueError("The TotalText dataset requires scipy and Pillow to be installed.")
    
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])
    gt_files = os.listdir(os.path.join(data_dir, "labels", "Train")) 

    data = []
    idx = 0
    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)

    for gt in gt_files:
        labels = loadmat(os.path.join(data_dir, "labels", "Train", gt))["gt"]
        img = Image.open(
            os.path.join(data_dir, "data", "Images", "Train", 
                         f"{gt.split('_')[-1].split('.')[0]}{'.JPG' if gt == 'gt_img61.mat' else '.jpg'}")
        )
        for l in labels:
            x, y, word = l[1][0], l[3][0], l[4][0]
            # For these images the annotations are not provided
            if word == "#":
                continue
            l, r = min(x), max(x)
            t, b = min(y), max(y)
            curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
            img.crop((l, t, r, b)).save(curr_remapped_path)
            data.append([curr_remapped_path, word])
            idx += 1
    return data

def _synthtext_preprocess_str(cfg):
    try: 
        from scipy.io import loadmat
    except ImportError:
        raise ValueError("The SynthText dataset requires scipy to be installed.")
    
    data_dir = os.path.join(cfg["root"], cfg["dataset_name"])

    zip_file = os.path.join(data_dir, "SynthText", "SynthText.zip")
    if os.path.exists(zip_file):
        extract_archive(from_path=zip_file, to_path=os.path.join(data_dir, "SynthText"), 
                        remove_finished=True)
    gt = loadmat(os.path.join(data_dir, "SynthText", "SynthText", "gt.mat"))

    data = []
    idx = 0
    remapped_imgs_path = os.path.join(data_dir, "remapped_imgs")
    os.makedirs(remapped_imgs_path, exist_ok=True)

    bboxes, imgs_paths, anns = gt["wordBB"], gt["imnames"], gt["txt"]
    for b, p, a in zip(bboxes, imgs_paths, anns):
        assert len(b) == len(p) == len(a)
        for bi, pi, ai in tqdm(zip(b, p, a)):
            if len(bi.shape) != 3:
                bi = bi.reshape(*bi.shape, 1)
            bi = bi.astype(np.int32)
            x, y = bi[0], bi[1]
            img = cv2.imread(os.path.join(data_dir, "SynthText", "SynthText", pi[0]))
            h, w, _ = img.shape
            ai_split = []
            for words in ai:
                ai_split.extend(words.split())
            for i in range(x.shape[-1]):
                curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
                t, b = max(0, min(y[:, i])), min(h, max(y[:, i])) 
                l, r = max(0, min(x[:, i])), min(w, max(x[:, i]))
                # Some annotations contain errors where this happens after clamping
                if t >= b or l >= r:
                    continue
                cv2.imwrite(curr_remapped_path, img[t:b, l:r])
                data.append([curr_remapped_path, ai_split[i].strip()])
                idx += 1
    return data

def _union14ml_preprocess_str(cfg):
    # TODO
    pass

def preprocess(cfg):
    fn_name = f"_{cfg['dataset_name']}_preprocess_{cfg['task']}"
    print(f"Looking for function {fn_name} . . .")
    preprocess_fn = getattr(sys.modules[__name__], fn_name, None)

    if preprocess_fn is None:
        raise ValueError(f"The function {fn_name} was not found. "
                         "The given dataset is not supported or you have not downloaded the dataset.")
    
    print("Found the function and now doing preprocessing . . .")
    return preprocess_fn(cfg)

# WildReceipt
# Synthetic Word Dataset (MJSynth/Syn90k)
# NAF
# FUNSD
# CTW1500
# COCO Text v2

# Modified from torchvision as some datasets cant pass the certificate validity check:
# https://github.com/pytorch/vision/blob/868a3b42f4bffe29e4414ad7e4c7d9d0b4690ecb/torchvision/datasets/utils.py#L27C1-L32C40
def urlretrieve(url, filename, chunk_size=1024 * 32, check_validity=True):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ctx = ssl.create_default_context()
    if not check_validity:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, context=ctx) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length, unit="B", unit_scale=True) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))

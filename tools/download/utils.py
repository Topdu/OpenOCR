import argparse
import urllib
import ssl
from tqdm import tqdm
import os
import json
import csv
import cv2
import xml.etree.ElementTree as ET

URLS = {
    "iiit": (
        {
            "str": [
                ("https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz", "data.tar.gz"),
            ]
        }, 
        False,
    ),
    "icdar13": (
        {
            "str": [
                ("https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task3_Images_GT.zip", "train_images_anns.zip"),
                ("https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_Images.zip", "test.zip"),
                ("https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_GT.txt", "test_anns.txt"),
            ]
        }, 
        False,
    ),
    "icdar15": (
        {
            "str": [
                ("https://rrc.cvc.uab.es/downloads/ch2_training_images.zip", "train_images.zip"),
                ("https://rrc.cvc.uab.es/downloads/ch2_training_localization_transcription_gt.zip", "train_localization_transcription.zip"),
                # ("https://rrc.cvc.uab.es/downloads/ch2_training_vocabularies_per_image.zip", "train_vocabularies.zip"),
                # ("https://rrc.cvc.uab.es/downloads/ch2_training_vocabulary.txt", "training_vocabulary.txt"),
                # ("https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip", "test_images.zip"),
                # ("https://rrc.cvc.uab.es/downloads/ch2_test_vocabularies_per_image.zip", "test_vocabularies_per_image.zip"),
                # ("https://rrc.cvc.uab.es/downloads/ch2_test_vocabulary.txt", "test_vocabulary.txt"),
            ]
        }, 
        False,
    ),
    "svt": (
        {
            "str": [
                ("http://www.iapr-tc11.org/dataset/SVT/svt.zip", "data.zip")
            ]
        },
        True
    ),
    "svtp": (
        {
            "str": [
                # The original page is not available, so we fall back to a paper that uses the dataset
                ("https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets/archive/refs/heads/master.zip", "data.zip")
            ]
        },
        True
    ),
    "cute": (
        {
            "str": [
                # We refer to these URLs as the original download link does not provide labels
                ("https://download.openmmlab.com/mmocr/data/mixture/ct80/timage.tar.gz", "data.tar.gz"),
                ("https://download.openmmlab.com/mmocr/data/1.x/recog/ct80/textrecog_test.json", "anns.json")
            ]
        },
        True
    ),
    "union-14m-l": (
        {
            "str": [
                ("https://drive.usercontent.google.com/download?id=18qbJ29K81Ub82bSTlSGG3O3fVVLjfVAu&confirm=t", "data.tar.gz")
            ]
        },
        True
    )
}

def get_dataset_info(args):
    urls_and_filenames, check_validity = URLS[args.dataset_name]
    urls, filename_paths, unpack_paths = [], [], []
    for u, f in urls_and_filenames[args.task]:
        urls.append(u)
        unpack_path = os.path.join(args.root, args.dataset_name, f.split(".")[0])
        unpack_paths.append(unpack_path)
        filename_paths.append(os.path.join(unpack_path, f))
    return urls, filename_paths, unpack_paths, check_validity


def _iiit_preprocess_str(args):
    try: 
        from scipy.io import loadmat
    except ImportError:
        raise ValueError("The IIIT dataset requires scipy to be installed.")
    
    data_path = os.path.join(args.root, args.dataset_name, "data", "IIIT5K")
    train_set = loadmat(os.path.join(data_path, "trainCharBound.mat"))["trainCharBound"][0]
    test_set = loadmat(os.path.join(data_path, "testCharBound.mat"))["testCharBound"][0]
    
    data = []
    for path, label, _ in train_set:
        path, label = path[0], label[0]
        if len(label) >= args.max_len:
            continue
        data.append(
            [os.path.join(data_path, path), str(label)]
        )

    for path, label, _ in test_set:
        path, label = path[0], label[0]
        if len(label) >= args.max_len:
            continue
        data.append(
            [os.path.join(data_path, path), str(label)]
        )

    return data

def _icdar13_preprocess_str(args):
    data_dir = os.path.join(args.root, args.dataset_name)
    train_images_path = os.path.join(data_dir, "train_images_anns")
    train_images_anns_path = os.path.join(train_images_path, "gt.txt")
    
    data = []  
    with open(train_images_anns_path, "r") as f:
        for l in f:
            path, label = l.strip().split(", ")
            label = label[1:-1]
            if len(label) >= args.max_len:
                continue
            data.append(
                [os.path.join(train_images_path, path), str(label)]
            )

    test_images_anns_path = os.path.join(data_dir, "test_anns", "test_anns.txt")
    with open(test_images_anns_path, "r") as f:
        for l in f:
            path, label = l.strip().split(", ")
            label = label[1:-1]
            if len(label) >= args.max_len:
                continue
            data.append(
                [os.path.join(data_dir, "test", path), str(label)]
            )
    return data

def _icdar15_preprocess_str(args):
    try:
        # We have to use PIL instead of opencv because of metadata rotations
        from PIL import Image
    except ImportError:
        raise ValueError("The ICDAR15 dataset requires scipy to be installed.")

    data_dir = os.path.join(args.root, args.dataset_name)
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
                if len(word) >= args.max_len or word == "###":
                    continue
                # LT, RT, RB, LB, WORD
                l, t, r, b = int(row[0]), int(row[1]), int(row[2]), int(row[5])
                curr_remapped_path = os.path.join(remapped_imgs_path, f"{idx}.jpg")
                img.crop((l, t, r, b)).save(curr_remapped_path)
                data.append([curr_remapped_path, word])
                idx += 1

    return data

def _svt_preprocess_str(args):
    data_dir = os.path.join(args.root, args.dataset_name, "data")

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
                        if len(label_text) >= args.max_len:
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

def _svtp_preprocess_str(args):
    data_dir = os.path.join(args.root, args.dataset_name, "data", "Case-Sensitive-Scene-Text-Recognition-Datasets-master", "svtp_test")
    
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

def _cute_preprocess_str(args):
    data_dir = os.path.join(args.root, args.dataset_name)

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

def _union_14m_l_preprocess_str(args):
    pass

PREPROCESS_FN = {
    "iiit": {
        "str":_iiit_preprocess_str,
    },
    "icdar13": {
        "str":_icdar13_preprocess_str,
    },
    "icdar15": {
        "str":_icdar15_preprocess_str,
    },
    "svt": {
        "str":_svt_preprocess_str,
    },
    "svtp": {
        "str":_svtp_preprocess_str,
    }, 
    "cute": {
        "str":_cute_preprocess_str,
    },
    "union-14m-l": {
        "str":_union_14m_l_preprocess_str,
    },
}

def get_preprocess_fn(args):
    if args.dataset_name not in PREPROCESS_FN:
        raise ValueError("The given dataset is not supported")
    
    dataset_path = os.path.join(args.root, args.dataset_name)
    if not os.path.isdir(dataset_path):
        raise ValueError(f"The given directory {dataset_path} does not exist. "
                         "If you have not downloaded the dataset use the tools/download_dataset.py script."
                         )
    
    preprocess_fn = PREPROCESS_FN[args.dataset_name][args.task]
    return preprocess_fn

def parse_args_download():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(URLS.keys()))
    parser.add_argument("--task", type=str, default="str", choices=("str")) # later add std, e2e
    args = parser.parse_args()
    return args

def parse_args_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(URLS.keys()))
    parser.add_argument("--task", type=str, default="str", choices=("str")) # later add std, e2e
    parser.add_argument("--max_len", type=int, default=800)
    args = parser.parse_args()
    return args

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

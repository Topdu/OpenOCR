import argparse
import urllib
import ssl
from tqdm import tqdm
import os

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
                ("https://rrc.cvc.uab.es/downloads/ch2_training_vocabularies_per_image.zip", "train_vocabularies.zip"),
                ("https://rrc.cvc.uab.es/downloads/ch2_training_vocabulary.txt", "training_vocabulary.txt"),
                ("https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip", "test_images.zip"),
                ("https://rrc.cvc.uab.es/downloads/ch2_test_vocabularies_per_image.zip", "test_vocabularies_per_image.zip"),
                ("https://rrc.cvc.uab.es/downloads/ch2_test_vocabulary.txt", "test_vocabulary.txt"),
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
                ("https://drive.usercontent.google.com/download?id=1LvkaRFXd7scdz7I-wqww8vRdqtLDDR46&confirm=t", "data.zip")
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
    pass

def _icdar15_preprocess_str(args):
    pass

def _svt_preprocess_str(args):
    pass

def _svtp_preprocess_str(args):
    pass

def _cute_preprocess_str(args):
    pass

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

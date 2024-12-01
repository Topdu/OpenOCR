import urllib
import ssl
from tqdm import tqdm
import os
import json
from tools.create_lmdb_dataset import createDataset
import datetime
import math
from PIL import Image
import io

def get_dataset_info(cfg):
    download_urls, filenames, processors, check_validity = cfg["download_links"], cfg["filenames"], cfg["processors"], cfg["check_validity"]
    urls, filename_paths = [], [] 
    for u, f in zip(download_urls, filenames):
        urls.append(u)
        filename_paths.append(os.path.join(cfg["root"], f))
    return urls, filename_paths, processors, check_validity

async def download_torrent(save_path, magnet_link, dataset_name):
    try: 
        from torrentp import TorrentDownloader
    except ImportError:
        raise ValueError(f"The {dataset_name} dataset requires torrentp to be installed.")
    torrent_file = TorrentDownloader(file_path=magnet_link, save_path=save_path)
    await torrent_file.start_download(download_speed=0, upload_speed=0)

# Adapted from:
# https://github.com/baudm/parseq/blob/main/tools/case_sensitive_str_datasets_converter.py
def process_case_sensitive_str_datasets_converter(cfg, file_list, mapping):
    p = []
    for f in file_list:
        for fi in os.listdir(f):
            potential_dir = os.path.join(f, fi)
            if os.path.isdir(potential_dir):
                p.append(os.path.join(f, fi))

    for pi in p:
        gt = []

        num_samples = len(list(filter(lambda x: x.endswith(".txt"), os.listdir(os.path.join(pi, "label")))))
        ext = os.listdir(os.path.join(pi, "IMG"))[0].split(".")[-1] 

        for i in range(1, num_samples + 1):
            img = os.path.join(pi, "IMG", f"{i}.{ext}")
            
            with open(os.path.join(pi, "label", f"{i}.txt"), "r") as f:
                label = f.readline()
            gt.append((img, label))

        curr_key = os.path.basename(pi)
        final_path = mapping[curr_key]
        createDataset(gt, final_path)

# Adapted from:
# https://github.com/baudm/parseq/blob/main/tools/coco_2_converter.py
def coco_text_converter(cfg, file_list, ann_file, output_dir, use_split_naming):
    img_dir = file_list[0]
    ann_path = os.path.join(cfg["root"], ann_file)
    coco_text = COCO_Text(ann_path)
    coco_text.createIndex()
    all_ann_ids = coco_text.getAnnIds()
    all_annotations = coco_text.loadAnns(all_ann_ids)

    if use_split_naming:
        data_list = {
            "train": [],
            "val": []
        }
    else:
        data_list = {
            "train": []
        }

    pad = 2
    for ann in all_annotations:
        text_label = ann.get("utf8_string", None)
        if (
            not text_label
            or ann['class'] != 'machine printed'
            or ann['language'] != 'english'
            or ann['legibility'] != 'legible'
        ):
            continue

        # Some labels and images with '#' in the middle are actually good, but some aren't, so we just filter them all.
        if text_label != '#' and '#' in text_label:
            continue

        # Some labels use '*' to denote unreadable characters
        if text_label.startswith('*') or text_label.endswith('*'):
            continue

        image_id = ann['image_id']
        image_info = coco_text.loadImgs([image_id])[0]
        img = Image.open(os.path.join(img_dir, image_info["file_name"]))
        src_w, src_h = img.size

        # L, T, W, H
        x, y, w, h = ann['bbox']
        x, y = max(0, math.floor(x) - pad), max(0, math.floor(y) - pad)
        w, h = math.ceil(w), math.ceil(h)
        x2, y2 = min(src_w, x + w + 2 * pad), min(src_h, y + h + 2 * pad)
        roi_img = img.crop((x, y, x2, y2))

        # NOTE: not happy with this as quality gets degraded
        buffer = io.BytesIO()
        roi_img.save(buffer, format="JPEG", quality=100)

        if use_split_naming:
            if image_info["set"] == "train":
                data_list["train"].append((buffer.getvalue(), text_label))
            else:
                data_list["val"].append((buffer.getvalue(), text_label))
        else:
            data_list["train"].append((buffer.getvalue(), text_label))

    if use_split_naming:
        createDataset(data_list["train"], os.path.join(output_dir, "train"))
        createDataset(data_list["val"], os.path.join(output_dir, "val"))
    else:
        createDataset(data_list["train"], output_dir)

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


# Taken from:
# https://github.com/andreasveit/coco-text/blob/master/coco_text.py
class COCO_Text:
    def __init__(self, annotation_file=None):
        """
        Constructor of COCO-Text helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = {}
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        self.val = []
        self.test = []
        self.train = []
        if not annotation_file == None:
            assert os.path.isfile(annotation_file), "file does not exist"
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        self.imgToAnns = {int(cocoid): self.dataset['imgToAnns'][cocoid] for cocoid in self.dataset['imgToAnns']}
        self.imgs      = {int(cocoid): self.dataset['imgs'][cocoid] for cocoid in self.dataset['imgs']}
        self.anns      = {int(annid): self.dataset['anns'][annid] for annid in self.dataset['anns']}
        self.cats      = self.dataset['cats']
        self.val       = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'val']
        self.test      = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'test']
        self.train     = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'train']
        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s'%(key, value))

    def filtering(self, filterDict, criteria):
        return [key for key in filterDict if all(criterion(filterDict[key]) for criterion in criteria)]

    def getAnnByCat(self, properties):
        """
        Get ann ids that satisfy given properties
        :param properties (list of tuples of the form [(category type, category)] e.g., [('readability','readable')] 
            : get anns for given categories - anns have to satisfy all given property tuples
        :return: ids (int array)       : integer array of ann ids
        """
        return self.filtering(self.anns, [lambda d, x=a, y=b:d[x] == y for (a,b) in properties])

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (list of tuples of the form [(category type, category)] e.g., [('readability','readable')] 
                : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = list(self.anns.keys())
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
            else:
                anns = list(self.anns.keys())
            anns = anns if len(catIds)  == 0 else list(set(anns).intersection(set(self.getAnnByCat(catIds)))) 
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if self.anns[ann]['area'] > areaRng[0] and self.anns[ann]['area'] < areaRng[1]]
        return anns

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = list(self.imgs.keys())
        else:
            ids = set(imgIds)
            if not len(catIds) == 0:
                ids  = ids.intersection(set([self.anns[annid]['image_id'] for annid in self.getAnnByCat(catIds)]))
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]
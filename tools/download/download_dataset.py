import os
import sys
import zipfile
import tarfile

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine import Config
from utility import ArgsParser
import download.utils
from torchvision.datasets.utils import extract_archive

def main(cfg):
    urls, filename_paths, processors, check_validity = download.utils.get_dataset_info(cfg)
    for url, filename_path, processor in zip(urls, filename_paths, processors):
        print(f"Downloading {filename_path} from {url} . . .")
        # download.utils.urlretrieve(url=url, filename=filename_path, check_validity=check_validity)

        # if filename_path.endswith(".zip"):
        #     with zipfile.ZipFile(filename_path, 'r') as zip_ref:
        #         file_list = zip_ref.namelist()
        # else:
        #     with tarfile.open(filename_path, 'r') as tar_ref:
        #         file_list = tar_ref.getnames()
        # file_list = [
        #     os.path.join(cfg["root"], name) for name in file_list 
        #     if name.endswith(os.sep) and name.count(os.sep) == 1
        # ]
        file_list=["data/train2014"]

        # extract_archive(from_path=filename_path, to_path=cfg["root"], remove_finished=True)

        if processor:
            if isinstance(processor, dict):
                processor_fn_name = next(iter(processor.keys()))
                processor_fn_kwargs = dict(list(processor.items())[1:])
            else:
                processor_fn_name = processor
                processor_fn_kwargs = {}
            processor_fn = getattr(download.utils, processor_fn_name)
            processor_fn(cfg, file_list, **processor_fn_kwargs)

    print("Downloads finished!")

if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

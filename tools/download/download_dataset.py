import os
import sys
import asyncio

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine import Config
from utility import ArgsParser
from download.utils import urlretrieve, download_torrent, get_dataset_info
from torchvision.datasets.utils import extract_archive

def main(cfg):
    urls, filename_paths, unpack_paths, check_validity = get_dataset_info(cfg)
    for url, filename_path, unpack_path in zip(urls, filename_paths, unpack_paths):
        print(f"Downloading {filename_path} from {url} . . .")
        ext = filename_path.split(".")[-1]
        if ext != "torrent":
            urlretrieve(url=url, filename=filename_path, check_validity=check_validity)
        else:
            asyncio.run(
                download_torrent(
                    magnet_link=filename_path, 
                    save_path=os.path.join(cfg.root, cfg.dataset_name), 
                    dataset_name=cfg.dataset_name
                )
            )

        if ext not in ["txt", "json", "torrent"]:
            extract_archive(from_path=filename_path, to_path=unpack_path, remove_finished=True)
    print("Downloads finished!")

if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

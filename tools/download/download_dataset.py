import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from download.utils import parse_args_download, urlretrieve, get_dataset_info
from torchvision.datasets.utils import extract_archive

def main():
    args = parse_args_download()
    urls, filename_paths, unpack_paths, check_validity = get_dataset_info(args)
    for url, filename_path, unpack_path in zip(urls, filename_paths, unpack_paths):
        print(f"Downloading {filename_path} from {url} . . .")
        urlretrieve(url=url, filename=filename_path, check_validity=check_validity)
        if not filename_path.endswith(".txt") and not filename_path.endswith(".json"):
            extract_archive(from_path=filename_path, to_path=unpack_path, remove_finished=True)
    print("Downloads finished!")

if __name__ == "__main__":
    main()
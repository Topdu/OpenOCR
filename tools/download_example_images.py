"""Download example images from ModelScope dataset for demo purposes."""
import os
from pathlib import Path
import shutil


def download_example_images():
    """Download example images from ModelScope dataset.

    Returns:
        Dict with paths to example image directories: {'ocr': path, 'doc': path, 'unirec': path}
    """
    # Will use dataset cache path folders directly
    subdirs = {}

    print(f'📥 Downloading example images...')

    download_success = False

    try:
        # Try ModelScope first (default)
        print('🌐 Trying ModelScope (China mirror) first...')
        try:
            # Download files directly from ModelScope dataset repository
            dataset_id = 'topdktu/openocr_test_images'

            # Try to get file list and download
            try:
                # This is a simplified approach - download via git clone or snapshot
                from modelscope.hub.snapshot_download import snapshot_download

                cache_path = snapshot_download(
                    repo_id=dataset_id,
                    repo_type='dataset',
                    cache_dir=str(Path.home() / '.cache' / 'openocr')
                )

                print(f'✅ Dataset downloaded from ModelScope to {cache_path}')

                # Use dataset cache path folders directly
                cache_dir = Path(cache_path)
                subdirs = {
                    'ocr': cache_dir / 'ocr',
                    'doc': cache_dir / 'doc',
                    'unirec': cache_dir / 'unirec'
                }

                # Verify folders exist and have images
                all_folders_valid = True
                for folder_name, folder_path in subdirs.items():
                    if folder_path.exists():
                        img_count = len([f for f in folder_path.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
                        if img_count > 0:
                            print(f'  ✓ Found {folder_name} folder with {img_count} images')
                        else:
                            print(f'  ⚠️  {folder_name} folder exists but has no images')
                            all_folders_valid = False
                    else:
                        print(f'  ⚠️  {folder_name} folder not found')
                        all_folders_valid = False

                if all_folders_valid:
                    download_success = True
                else:
                    print('⚠️  ModelScope download incomplete, trying HuggingFace...')
                    subdirs = {}

            except Exception as e:
                print(f'⚠️  ModelScope snapshot download failed: {e}')
                print('   Trying HuggingFace...')

        except ImportError:
            print('⚠️  modelscope not installed. Install with: pip install modelscope')
            print('   Trying HuggingFace...')
        except Exception as e:
            print(f'⚠️  ModelScope download failed: {e}')
            print('   Trying HuggingFace...')

        if not download_success:
            # Try HuggingFace
            print('🌐 Using HuggingFace...')
            try:
                from huggingface_hub import snapshot_download

                # Download entire dataset
                dataset_path = snapshot_download(
                    repo_id='topdu/openocr_test_images',
                    repo_type='dataset',
                    cache_dir=str(Path.home() / '.cache' / 'openocr')
                )

                print(f'✅ Dataset downloaded to {dataset_path}')

                # Use dataset cache path folders directly
                cache_dir = Path(dataset_path)
                subdirs = {
                    'ocr': cache_dir / 'ocr',
                    'doc': cache_dir / 'doc',
                    'unirec': cache_dir / 'unirec'
                }

                # Verify folders exist and have images
                all_folders_valid = True
                for folder_name, folder_path in subdirs.items():
                    if folder_path.exists():
                        img_count = len([f for f in folder_path.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
                        if img_count > 0:
                            print(f'  ✓ Found {folder_name} folder with {img_count} images')
                        else:
                            print(f'  ⚠️  {folder_name} folder exists but has no images')
                            all_folders_valid = False
                    else:
                        print(f'  ⚠️  {folder_name} folder not found')
                        all_folders_valid = False

                if all_folders_valid:
                    download_success = True

            except ImportError:
                print('⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub')
            except Exception as e:
                print(f'⚠️  HuggingFace download failed: {e}')

        # Try GitHub releases as fallback for OCR examples only
        if not download_success:
            print('🌐 Trying GitHub releases as fallback for OCR examples...')
            try:
                import urllib.request
                import tarfile
                import tempfile

                ocr_url = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/OCR_e2e_img.tar'

                # Use temp directory for download
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    tar_path = temp_path / 'OCR_e2e_img.tar'

                    print(f'  Downloading from {ocr_url}...')
                    urllib.request.urlretrieve(ocr_url, str(tar_path))

                    print(f'  Extracting...')
                    with tarfile.open(str(tar_path), 'r') as tar:
                        tar.extractall(path=str(temp_path))

                    # Move to cache directory
                    cache_base = Path.home() / '.cache' / 'openocr' / 'openocr_examples'
                    cache_base.mkdir(parents=True, exist_ok=True)

                    # Copy extracted files to cache
                    ocr_source = temp_path / 'OCR_e2e_img'
                    ocr_target = cache_base / 'ocr'
                    if ocr_source.exists():
                        if ocr_target.exists():
                            shutil.rmtree(str(ocr_target))
                        shutil.copytree(str(ocr_source), str(ocr_target))

                    # Set subdirs for GitHub download
                    subdirs = {
                        'ocr': ocr_target,
                        'doc': cache_base / 'doc',
                        'unirec': cache_base / 'unirec'
                    }

                    # Create empty directories for doc and unirec if they don't exist
                    for key in ['doc', 'unirec']:
                        subdirs[key].mkdir(parents=True, exist_ok=True)

                    print(f'  ✓ OCR example images downloaded from GitHub to cache')
                    download_success = True

            except Exception as e:
                print(f'⚠️  GitHub download failed: {e}')

        if download_success:
            print(f'✅ Example images ready!')
        else:
            print('⚠️  Could not download example images automatically.')

    except Exception as e:
        print(f'❌ Download failed: {e}')

    finally:
        # Verify directories
        if subdirs:
            print('\n📝 Example image directories:')
            for name, subdir in subdirs.items():
                if subdir.exists():
                    if not any(subdir.iterdir()):
                        print(f'   ⚠️  {name}: No images found in {subdir}')
                        print(f'      You can manually add example images to this directory.')
                    else:
                        img_count = len(list(subdir.glob('*.[jp][pn]g')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.bmp')))
                        print(f'   ✓ {name}: {img_count} images found in {subdir}')
                else:
                    print(f'   ⚠️  {name}: Directory not found at {subdir}')
        else:
            print('\n⚠️  No example image directories available')

    return {k: str(v) for k, v in subdirs.items()}


def get_example_images_path(demo_type='ocr'):
    """Get the path to example images for a specific demo type.

    Args:
        demo_type: Type of demo ('ocr', 'doc', or 'unirec')

    Returns:
        Path to example images directory
    """
    # Download and get paths from cache
    print(f'Getting example images for {demo_type}...')
    paths = download_example_images()

    # Return the path for the requested demo type
    if demo_type in paths:
        return paths[demo_type]
    else:
        print(f'⚠️  Unknown demo type: {demo_type}')
        return paths.get('ocr', '')


if __name__ == '__main__':
    # Test download
    import argparse

    parser = argparse.ArgumentParser(description='Download example images for OpenOCR demos')

    args = parser.parse_args()

    paths = download_example_images()

    print('\n📁 Example image directories:')
    for demo_type, path in paths.items():
        print(f'  {demo_type}: {path}')

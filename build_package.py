#!/usr/bin/env python
"""
Build script for OpenOCR package
This script reorganizes the project structure and builds the wheel package
"""

import shutil
import subprocess
import sys
import os
from pathlib import Path


def create_package_structure():
    """Create openocr package structure and copy necessary files"""

    # Get current directory
    root_dir = Path(__file__).parent.absolute()
    openocr_dir = root_dir / 'build/openocr'

    print(f'Root directory: {root_dir}')
    print(f'Target openocr directory: {openocr_dir}')

    # Remove existing openocr directory if it exists
    if openocr_dir.exists():
        print(f'Removing existing {openocr_dir}...')
        shutil.rmtree(openocr_dir)

    # Create openocr directory
    print(f'Creating {openocr_dir}...')
    os.makedirs(openocr_dir, exist_ok=True)
    # List of directories to copy
    dirs_to_copy = ['configs', 'docs', 'opendet', 'openrec', 'tools']

    # Copy directories
    for dir_name in dirs_to_copy:
        src = root_dir / dir_name
        dst = openocr_dir / dir_name
        if src.exists():
            print(f'Copying {dir_name}...')
            shutil.copytree(src, dst)
        else:
            print(f'Warning: {dir_name} not found, skipping...')

    # List of files to copy
    files_to_copy = ['__init__.py', 'openocr.py', 'demo_gradio.py', 'demo_opendoc.py', 'demo_unirec.py']
    # Copy files
    for file_name in files_to_copy:
        src = root_dir / file_name
        dst = openocr_dir / file_name
        if src.exists():
            print(f'Copying {file_name}...')
            shutil.copy2(src, dst)
        else:
            print(f'Warning: {file_name} not found, skipping...')
    files_to_copy = ['setup.py', 'MANIFEST.in', 'LICENSE', 'pyproject.toml', 'QUICKSTART.md']
    for file_name in files_to_copy:
        src = root_dir / file_name
        dst = root_dir / 'build' / file_name
        if src.exists():
            print(f'Copying {file_name}...')
            shutil.copy2(src, dst)
        else:
            print(f'Warning: {file_name} not found, skipping...')



    # Verify __init__.py was copied and contains __version__
    init_file = openocr_dir / '__init__.py'
    if init_file.exists():
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if '__version__' not in content:
                print('Warning: __init__.py does not contain __version__, this may cause issues')
    else:
        print('ERROR: __init__.py was not copied successfully!')

    print('Package structure created successfully!')
    return openocr_dir


def build_wheel():
    """Build the wheel package"""

    root_dir = Path(__file__).parent.absolute()
    build_root = root_dir / 'build'

    print('\n' + '='*60)
    print('Building wheel package...')
    print('='*60 + '\n')

    # Clean previous builds
    dist_dir = build_root / 'dist'
    build_dir = build_root / 'build'
    egg_info_dir = build_root / 'openocr_python.egg-info'

    for dir_path in [dist_dir, build_dir, egg_info_dir]:
        if dir_path.exists():
            print(f'Cleaning {dir_path}...')
            shutil.rmtree(dir_path)

    # Build wheel from build directory
    print('\nRunning: python setup.py sdist bdist_wheel')
    result = subprocess.run(
        [sys.executable, 'setup.py', 'sdist', 'bdist_wheel'],
        cwd=build_root,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print('STDERR:', result.stderr)

    if result.returncode == 0:
        print('\n' + '='*60)
        print('Build successful!')
        print('='*60)
        print(f'\nWheel package created in: {dist_dir}')

        # List created files
        if dist_dir.exists():
            print('\nCreated files:')
            for file in dist_dir.iterdir():
                print(f'  - {file.name}')
        return True
    else:
        print('\n' + '='*60)
        print('Build failed!')
        print('='*60)
        return False


def main():
    """Main function"""

    print('='*60)
    print('OpenOCR Package Build Script')
    print('='*60 + '\n')

    try:
        # Step 1: Create package structure
        print('Step 1: Creating package structure...')
        create_package_structure()

        # Step 2: Build wheel
        print('\nStep 2: Building wheel package...')
        success = build_wheel()

        if success:
            print('\n' + '='*60)
            print('All steps completed successfully!')
            print('='*60)
            print('\nTo install the package, run:')
            print(f'  pip install build/dist/openocr_python-*.whl')
            print('\nAfter installation, you can use:')
            print('  openocr --help')
        else:
            print('\nBuild failed. Please check the error messages above.')
            sys.exit(1)

    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

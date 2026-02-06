#!/usr/bin/env python
"""
OpenOCR - Accurate and Efficient General OCR System
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), 'openocr', filename)
    if os.path.exists(filepath):
        with open(filepath, encoding='utf-8') as f:
            return f.read()
    return ''

# Get version
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'openocr', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

setup(
    name='openocr-python',
    version=get_version(),
    description='Accurate and Efficient General OCR System',
    long_description=read_file('QUICKSTART.md'),
    long_description_content_type='text/markdown',
    author='OCR Team, FVL Lab',
    author_email='784990967@qq.com',
    url='https://github.com/Topdu/OpenOCR',
    license='Apache License 2.0',

    # Package configuration
    packages=find_packages(),
    include_package_data=True,

    # Python version requirement
    python_requires='>=3.8',

    # Dependencies
    install_requires=[
        'numpy==1.26.4',
        'opencv-python<=4.12.0',
        'Pillow<=9.5.0',
        'pyyaml<=6.0.0',
        'tqdm<=4.65.0',
        'rapidfuzz<=2.0.0',
        'pyclipper<=1.3.0',
        'pydantic',
        'shapely',
    ],

    # Optional dependencies
    extras_require={
        'onnx': [
            'onnxruntime>=1.12.0',
            'onnxruntime-gpu>=1.12.0',
        ],
        'pytorch': [
            'torch>=1.10.0',
            'torchvision>=0.11.0',
        ],
        'gradio': [
            'gradio==4.20.0',
            'onnxruntime>=1.12.0',
            'onnxruntime-gpu>=1.12.0',
        ],
        'huggingface': [
            'huggingface-hub>=0.16.0',
        ],
        'modelscope': [
            'modelscope>=1.9.0',
        ],
        'all': [
            'onnxruntime>=1.12.0',
            'torch>=1.10.0',
            'torchvision>=0.11.0',
            'gradio==4.20.0',
            'huggingface-hub>=0.16.0',
            'modelscope>=1.9.0',
        ],
    },

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'openocr=openocr.openocr:main',
        ],
    },

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Keywords
    keywords='ocr, optical character recognition, text detection, text recognition, document parsing, deep learning, computer vision',

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/Topdu/OpenOCR/issues',
        'Source': 'https://github.com/Topdu/OpenOCR',
        'Documentation': 'https://github.com/Topdu/OpenOCR/tree/main/docs',
    },
    # Zip safe
    zip_safe=False,
)

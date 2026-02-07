from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# from .tools.infer_e2e import OpenOCRE2E, OpenDetector, OpenRecognizer
# from .tools.infer_unirec_onnx import UniRecONNX
# from .tools.infer_doc_onnx import OpenDocONNX
from .openocr import OpenOCR, main

__version__ = '0.1.0.dev'

# Lazy import for demo interfaces to avoid initialization on import
def launch_openocr_demo(*args, **kwargs):
    """Launch Gradio OCR demo"""
    from .demo_gradio import launch_demo
    return launch_demo(*args, **kwargs)

def launch_unirec_demo(*args, **kwargs):
    """Launch UniRec demo"""
    from .demo_unirec import launch_demo
    return launch_demo(*args, **kwargs)

def launch_opendoc_demo(*args, **kwargs):
    """Launch OpenDoc demo"""
    from .demo_opendoc import launch_demo
    return launch_demo(*args, **kwargs)

__all__ = [
    'OpenOCRE2E',
    'OpenDetector',
    'OpenRecognizer',
    'UniRecONNX',
    'OpenDocONNX',
    'OpenOCR',
    'main',
    'launch_openocr_demo',
    'launch_unirec_demo',
    'launch_opendoc_demo',
]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.infer_e2e import OpenOCR, OpenDetector, OpenRecognizer

"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from utils.config import Config
from utils.visualize import decode_labels
from utils.image_reader import ImageReader, prepare_label
a = b + Config

shengchao = 5
jiangkai = 6
cnacan = 7

wenhe = 8
jiangkai = 9

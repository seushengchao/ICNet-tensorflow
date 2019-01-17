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

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")
    
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'others'])
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()


shengchao = 1


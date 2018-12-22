# Copyright 2018 CGY. All Rights Reserved.

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib

import depthmodel

def main(args):
    sess, params, model = depthmodel.init(uploads_path=args.dataset, test_file_list=args.filenames_file, checkpoint_path=args.checkpoint_path)
    depthmodel.test(sess, params, model, data_path=args.dataset, filenames_file=args.filenames_file, output_directory=args.output_directory, file_name_pr='result')

    exit()

"""
CUDA_VISIBLE_DEVICES=12 python main.py --dataset=/home/raw_datasets/3D/data/images/ \
--filenames_file=/home/raw_datasets/3D/data/test_files_eigen.txt \
--output_directory=/home/raw_datasets/3D/data/results/images \
--checkpoint_path=/home/raw_datasets/3D/depth/models/model_city2kitti.meta
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')

    args = parser.parse_args()

    main(args)
    


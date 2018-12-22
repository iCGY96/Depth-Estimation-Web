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

from depthmodel.depth_model import *
from depthmodel.dataloader import *
from depthmodel.average_gradients import *

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(sess, params, model, data_path, filenames_file, output_directory, file_name_pr):
    """Test function."""

    dataloader = Dataloader(data_path=data_path, filenames_file=filenames_file, params=params, dataset='kitti', mode='test')
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model.left = left
    model.right = right

    num_test_samples = count_text_lines(filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_left_est[0])
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

    print('done.')

    print('writing disparities.')
    # output_directory = output_directory
    for i in range(disparities.shape[0]):
        result = (disparities[i] - np.min(disparities[i])) / (np.max(disparities[i]) - np.min(disparities[i]))
        if os.path.exists(os.path.join(output_directory, file_name_pr + '.png')):
            os.remove(os.path.join(output_directory, file_name_pr + '.png'))
        matplotlib.image.imsave(os.path.join(output_directory, file_name_pr + '.png'), result)
        result_pp = (disparities_pp[i] - np.min(disparities_pp[i])) / (np.max(disparities_pp[i]) - np.min(disparities_pp[i]))
        if os.path.exists(os.path.join(output_directory, file_name_pr + '_pp.png')):
            os.remove(os.path.join(output_directory, file_name_pr + '_pp.png'))
        matplotlib.image.imsave(os.path.join(output_directory, file_name_pr + '_pp.png'), result_pp)
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)

    print('done.')


def init(uploads_path='/path/to/uploads/', test_file_list='/path/to/test_files_eigen.txt', checkpoint_path='/path/to/model'):
    """ The params for the depth model """
    params = depthmodel_parameters(
        encoder='vgg', # vgg or resnet50
        height=256, # input height
        width=512, # input width
        batch_size=8, # batch size
        num_threads=8, # number of threads to use for data loading
        num_epochs=50, # number of epochs
        do_stereo=False, # if set, will train the stereo model
        wrap_mode='border', # 'bilinear sampler wrap mode, edge or border'
        use_deconv=False, # 'if set, will use transposed convolutions'
        alpha_image_loss=0.85, # weight between SSIM and L1 in the image loss
        disp_gradient_loss_weight=0.1, # disparity smoothness weigth
        lr_loss_weight=1.0, # left-right consistency weight
        full_summary=False) # if set, will keep more data for each summary. Warning: the file can become very large

    dataloader = Dataloader(data_path=uploads_path, 
    filenames_file=test_file_list, 
    params=params, dataset='kitti', mode='test')
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MonodepthModel(params, 'test', left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = checkpoint_path.split(".")[0]
    # print(restore_path)
    train_saver.restore(sess, restore_path)

    return sess, params, model
    


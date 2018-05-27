#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def single_compress(config, args):
    start = time.time()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = np.array([args.image_path]), np.array([args.semantic_map_path])
    else:
        paths = np.array([args.image_path])

    gan = Model(config, paths, name='single_compress', dataset=args.dataset, evaluate=True)
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_init = {gan.path_placeholder: paths,
                          gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_init = {gan.path_placeholder: paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(gan.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_init)
        eval_dict = {gan.training_phase: False, gan.handle: handle}

        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(directories.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        Utils.single_plot(0, 0, sess, gan, handle, save_path, config, single_compress=True)
        print('Reconstruction saved to', save_path)

    return


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-i", "--image_path", help="path to image to compress", type=str)
    parser.add_argument("-sm", "--semantic_map_path", help="path to corresponding semantic map", type=str)
    parser.add_argument("-o", "--output_path", help="path to output image", type=str)
    parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    args = parser.parse_args()

    # Launch training
    single_compress(config_test, args)

if __name__ == '__main__':
    main()

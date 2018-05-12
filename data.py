#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import directories


class Data(object):

    @staticmethod
    def load_dataframe(filename):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)

        return df['path'].values

    def preprocess_inference(image_path, resize=(32,32)):
        # Preprocess individual images during inference

        image_path = tf.squeeze(image_path)
        image = tf.image.decode_png(tf.read_file(image_path))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_images(image, size=resize)

        return image

    @staticmethod
    def load_dataset(filenames, batch_size, test=False, augment=False, downsample=False, multiscale=False):

        # Consume image data
        def _augment(image):
            # On-the-fly data augmentation
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.image.random_flip_left_right(image)
            image = random_rotation(image, 0.04, crop=True) # radians

            return image

        def _parser(image_path):
            def _aspect_preserving_width_resize(image, width=512):
                height_i = tf.shape(image)[0]
                width_i = tf.shape(image)[1]
                ratio = tf.to_float(width_i) / tf.to_float(height_i)
                new_height = tf.to_int32(tf.to_float(height_i) / ratio)
                new_height = new_height - tf.floormod(new_height, 16)
                    
                return tf.image.resize_images(image, [new_height,width])

            if multiscale is True:
                scales = [1,2,4]
                pyramid = list()
                for scale in scales:
                    im = tf.image.decode_jpeg(tf.read_file(image_path), channels=3, ratio=scale)
                    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
                    im = 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
                    # im.set_shape([512,1024,3])
                    pyramid.append(im)  # first element of the list should be the original image

                return pyramid

            else:
                im = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
                im = tf.image.convert_image_dtype(im, dtype=tf.float32)
                im = 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
                # im = _aspect_preserving_width_resize(im)
                im.set_shape([512,1024,3])

                return im

        dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(_parser)
        dataset = dataset.shuffle(buffer_size=8)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()

        return dataset

    @staticmethod
    def load_inference(filenames, labels, batch_size, resize=(32,32)):

        # Single image estimation over multiple stochastic forward passes

        def _preprocess_inference(image_path, label, resize=(32,32)):
            # Preprocess individual images during inference
            image_path = tf.squeeze(image_path)
            image = tf.image.decode_png(tf.read_file(image_path))
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.image.resize_images(image, size=resize)

            return image, label

        dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_preprocess_inference)
        dataset = dataset.batch(batch_size)
        
        return dataset


#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import directories

class Data(object):

    @staticmethod
    def load_dataframe(filename, load_semantic_maps=False):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)

        if load_semantic_maps:
            return df['path'].values, df['semantic_map_path'].values
        else:
            return df['path'].values

    @staticmethod
    def load_dataset(image_paths, batch_size, test=False, augment=False, downsample=False,
            training_dataset='cityscapes', use_conditional_GAN=False, **kwargs):

        def _augment(image):
            # On-the-fly data augmentation
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.image.random_flip_left_right(image)

            return image

        def _parser(image_path, semantic_map_path=None):

            def _aspect_preserving_width_resize(image, width=512):
                height_i = tf.shape(image)[0]
                # width_i = tf.shape(image)[1]
                # ratio = tf.to_float(width_i) / tf.to_float(height_i)
                # new_height = tf.to_int32(tf.to_float(height_i) / ratio)
                new_height = height_i - tf.floormod(height_i, 16)
                return tf.image.resize_image_with_crop_or_pad(image, new_height, width)

            def _image_decoder(path):
                im = tf.image.decode_png(tf.read_file(path), channels=3)
                im = tf.image.convert_image_dtype(im, dtype=tf.float32)
                return 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
                    
            image = _image_decoder(image_path)

            # Explicitly set the shape if you want a sanity check
            # or if you are using your own custom dataset, otherwise
            # the model is shape-agnostic as it is fully convolutional

            # im.set_shape([512,1024,3])  # downscaled cityscapes

            if use_conditional_GAN:
                # Semantic map only enabled for cityscapes
                semantic_map = _image_decoder(semantic_map_path)           

            if training_dataset == 'ADE20k':
                image = _aspect_preserving_width_resize(image)
                # im.set_shape([None,512,3])

            if use_conditional_GAN:
                if training_dataset == 'ADE20k':
                    raise NotImplementedError('Conditional generation not implemented for ADE20k dataset.')
                return image, semantic_map
            else:
                return image
            

        print('Training on', training_dataset)

        if use_conditional_GAN:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, kwargs['semantic_map_paths']))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        dataset = dataset.map(_parser)
        dataset = dataset.shuffle(buffer_size=8)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()

        return dataset

    @staticmethod
    def load_cGAN_dataset(image_paths, semantic_map_paths, batch_size, test=False, augment=False, downsample=False,
            training_dataset='cityscapes'):
        """
        Load image dataset with semantic label maps for conditional GAN
        """ 

        def _parser(image_path, semantic_map_path):
            def _aspect_preserving_width_resize(image, width=512):
                # If training on ADE20k
                height_i = tf.shape(image)[0]
                new_height = height_i - tf.floormod(height_i, 16)
                    
                return tf.image.resize_image_with_crop_or_pad(image, new_height, width)

            def _image_decoder(path):
                im = tf.image.decode_png(tf.read_file(image_path), channels=3)
                im = tf.image.convert_image_dtype(im, dtype=tf.float32)
                return 2 * im - 1 # [0,1] -> [-1,1] (tanh range)


            image, semantic_map = _image_decoder(image_path), _image_decoder(semantic_map_path)
            
            print('Training on', training_dataset)
            if training_dataset is 'ADE20k':
                image = _aspect_preserving_width_resize(image)
                semantic_map = _aspect_preserving_width_resize(semantic_map)

            # im.set_shape([512,1024,3])  # downscaled cityscapes

            return image, semantic_map

        dataset = tf.data.Dataset.from_tensor_slices(image_paths, semantic_map_paths)
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

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_preprocess_inference)
        dataset = dataset.batch(batch_size)
        
        return dataset


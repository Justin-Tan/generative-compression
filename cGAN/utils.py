# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session

import tensorflow as tf
import numpy as np
import os, time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import directories

class Utils(object):
    
    @staticmethod
    def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def residual_block(x, n_filters, kernel_size=3, strides=1, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        strides = [1,1]
        identity_map = x

        p = int((kernel_size-1)/2)
        res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = actv(tf.contrib.layers.instance_norm(res))

        res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = tf.contrib.layers.instance_norm(res)

        assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
        out = tf.add(res, identity_map)

        return out

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle, start_time, epoch, name, G_loss_best, D_loss_best):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_test = {model.training_phase: False, model.handle: train_handle}

        try:
            G_loss, D_loss, summary = sess.run([model.G_loss, model.D_loss, model.merge_op], feed_dict=feed_dict_test)
            model.train_writer.add_summary(summary)
        except tf.errors.OutOfRangeError:
            G_loss, D_loss = float('nan'), float('nan')

        if G_loss < G_loss_best and D_loss < D_loss_best:
            G_loss_best, D_loss_best = G_loss, D_loss
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, '{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 5 == 0 and epoch > 5:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, '{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Generator Loss: {:.3f} | Discriminator Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, G_loss, D_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return G_loss_best, D_loss_best

    @staticmethod
    def single_plot(epoch, global_step, sess, model, handle, name, config):

        real = model.example
        gen = model.reconstruction

        # Generate images from noise, using the generator network.
        r, g = sess.run([real, gen], feed_dict={model.training_phase:True, model.handle: handle})

        images = list()

        for im, imtype in zip([r,g], ['real', 'gen']):
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            im = im[:,:,:3]
            images.append(im)

            # Uncomment to plot real and generated samples separately
            # f = plt.figure()
            # plt.imshow(im)
            # plt.axis('off')
            # f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}.pdf".format(directories.samples, name, epoch,
            #                     global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            # plt.gcf().clear()
            # plt.close(f)

        comparison = np.hstack(images)
        f = plt.figure()
        plt.imshow(comparison)
        plt.axis('off')
        f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}_comparison.pdf".format(directories.samples, name, epoch,
            global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
        plt.gcf().clear()
        plt.close(f)


    @staticmethod
    def weight_decay(weight_decay, var_label='DW'):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'{}'.format(var_label)) > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(weight_decay, tf.add_n(costs))


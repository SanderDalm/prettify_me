from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from glob import glob

import models
import numpy as np
import tensorflow as tf


#""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
args = parser.parse_args()

dataset = args.dataset
load_size = args.load_size
crop_size = args.crop_size
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr


#""" graph """
# models
generator_a2b = partial(models.generator, scope='a2b')
generator_b2a = partial(models.generator, scope='b2a')
discriminator_a = partial(models.discriminator, scope='a')
discriminator_b = partial(models.discriminator, scope='b')

# Placeholders
a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
# a2b_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
# b2a_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

# Generator outputs
a2b = generator_a2b(a_real)
a2b2a = generator_b2a(a2b)

b2a = generator_b2a(b_real)
b2a2b = generator_a2b(b2a)


# Discriminator outputs
# a2b
a_logit = discriminator_a(a_real)
b2a_logit = discriminator_a(b2a)

# b2a
b_logit = discriminator_b(b_real)
a2b_logit = discriminator_b(a2b)

# Sample ??
# b2a_sample_logit = discriminator_a(b2a_sample)
# a2b_sample_logit = discriminator_b(a2b_sample)

# Generator losses
g_loss_a2b = tf.losses.mean_squared_error(a2b_logit, tf.ones_like(a2b_logit))
g_loss_b2a = tf.losses.mean_squared_error(b2a_logit, tf.ones_like(b2a_logit))
cyc_loss_a = tf.losses.absolute_difference(a_real, a2b2a)
cyc_loss_b = tf.losses.absolute_difference(b_real, b2a2b)
g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a * 10.0 + cyc_loss_b * 10.0

# Discriminator a losses
d_loss_a_real = tf.losses.mean_squared_error(a_logit, tf.ones_like(a_logit))
d_loss_b2a = tf.losses.mean_squared_error(b2a_logit, tf.zeros_like(b2a_logit))
d_loss_a = d_loss_a_real + d_loss_b2a

# Discriminator b losses
d_loss_b_real = tf.losses.mean_squared_error(b_logit, tf.ones_like(b_logit))
d_loss_a2b = tf.losses.mean_squared_error(a2b_logit, tf.zeros_like(a2b_logit))
d_loss_b = d_loss_b_real + d_loss_a2b

# Optimization
t_var = tf.trainable_variables()
d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)


# """ train """
# ''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#''' saver '''
saver = tf.train.Saver(max_to_keep=5)

#'''train'''
for i in range(1000):

    batchgen = None
    a_real_batch, b_real_batch = batchgen.generate_batch()

    # train G a+b
    g_summary_opt, _ = sess.run([g_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # train D a
    d_summary_a_opt, _ = sess.run([d_a_train_op], feed_dict={a_real: a_real_batch})

    # train D b
    d_summary_b_opt, _ = sess.run([d_b_train_op], feed_dict={b_real: b_real_batch})

    # save
    if i + 1 % 1000 == 0:
        save_path = saver.save(sess, 'path')
        print('Model saved in file: % s' % save_path)

    # sample
    if i + 1 % 100 == 0:

        a2b_output, a2b2a_output, b2a_output, b2a2b_output = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a_real, b_real: b_real})
        sample = np.concatenate((a2b_output, a2b2a_output, b2a_output, b2a2b_output), axis=0)
        save_dir = '/outputs/'
        # TO DO: save
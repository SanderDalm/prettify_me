from functools import partial

import numpy as np
from scipy.misc import imsave
import tensorflow as tf
from glob import glob

from cycle_gan.two_class_batch_generator import TwoClassBatchGenerator
import cycle_gan.nn as nn
import config

crop_size = 100
lr = .0001
batch_size = 16

########################
# Batch gen
########################
file_list_a = glob(config.datadir+'/UTKFace/*')
mislabeled = []
for x in file_list_a:
    try:
        age = x.split('_')[-4].split('/')[-1]
        int(age)
    except:
        mislabeled.append(x)

file_list_a = [x for x in file_list_a if x not in mislabeled]


file_list_a = [x for x in file_list_a if 18 < int(x.split('_')[-4].split('/')[-1]) < 40] # young
file_list_a = [x for x in file_list_a if x.split('_')[-2]=='2']  # asian
file_list_a = [x for x in file_list_a if x.split('_')[-3]=='1'] # women


file_list_b = glob(config.datadir+'/composites/*')

batchgen = TwoClassBatchGenerator(file_list_a=file_list_a, file_list_b=file_list_b, height=crop_size, width=crop_size)


#""" graph """
# models
generator_a2b = partial(nn.generator, scope='a2b')
generator_b2a = partial(nn.generator, scope='b2a')
discriminator_a = partial(nn.discriminator, scope='a')
discriminator_b = partial(nn.discriminator, scope='b')

# Placeholders
a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

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


# Generator losses
# Domain loss
g_loss_a2b = tf.losses.mean_squared_error(a2b_logit, tf.ones_like(a2b_logit))
g_loss_b2a = tf.losses.mean_squared_error(b2a_logit, tf.ones_like(b2a_logit))

# Cycle loss
cyc_loss_a = tf.losses.absolute_difference(a_real, a2b2a)
cyc_loss_b = tf.losses.absolute_difference(b_real, b2a2b)

# Sum loss
g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a * 10.0 + cyc_loss_b * 10.0

# Discriminator losses
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

init_op = tf.global_variables_initializer()
sess.run(init_op)

#''' saver '''
saver = tf.train.Saver(max_to_keep=None)

#'''train'''
for i in range(1, 100001):

    a_real_batch, b_real_batch = batchgen.generate_batch(batch_size)
    a_real_batch, b_real_batch = (a_real_batch * 2) - 1, (b_real_batch * 2) - 1

    # train G a+b
    g_summary_opt = sess.run([g_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # train D a
    d_summary_a_opt = sess.run([d_a_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # train D b
    d_summary_b_opt = sess.run([d_b_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # save sample
    if i % 1000 == 0:
        a2b_output, a2b2a_output, b2a_output, b2a2b_output = sess.run([a2b, a2b2a, b2a, b2a2b],
                                                                      feed_dict={a_real: a_real_batch[0:1],
                                                                                 b_real: b_real_batch[0:1]})
        sample = np.concatenate(
            [a_real_batch[0:1], a2b_output, a2b2a_output, b_real_batch[0:1], b2a_output,
             b2a2b_output], axis=1)
        sample = sample.reshape(crop_size * 6, crop_size, 3)
        save_path = 'cycle_gan/samples/sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 10000 == 0:
        save_path = saver.save(sess, 'cycle_gan/models/nn_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))
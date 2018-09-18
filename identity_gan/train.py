from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import importlib

from cycle_gan.men_women_batch_generator import Men_Women_BatchGenerator
import cycle_gan.nn as nn
import config

##################
# session
##################
sesssion_config = tf.ConfigProto(allow_soft_placement=True)
sesssion_config.gpu_options.allow_growth = True
sess = tf.Session(config=sesssion_config)

##############################
# Facenet for identity vector
##############################
network = importlib.import_module('identity_gan.models.inception_resnet_v1')

def get_identity_vector(img, reuse):
    with tf.variable_scope('facenet', reuse=reuse):
        prelogits, endpoints = network.inference(images=img, keep_probability=1, phase_train=False,
                                            bottleneck_layer_size=512, weight_decay=0.0)


        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        return embeddings

################################
# Cycle gan
################################
crop_size = 152
lr = .0001
batch_size = 16
batchgen = Men_Women_BatchGenerator(config.datadir, height=crop_size, width=crop_size)

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

# Identity loss
identity_a_before = get_identity_vector(a_real, reuse=False)
identity_a_after = get_identity_vector(a2b, reuse=True)
identity_diff_a = tf.norm(identity_a_before - identity_a_after, axis=1)

identity_b_before = get_identity_vector(b_real, reuse=True)
identity_b_after = get_identity_vector(b2a, reuse=True)
identity_diff_b = tf.norm(identity_b_before - identity_b_after, axis=1)

identity_loss = tf.reduce_mean(identity_diff_a + identity_diff_b)

# Sum loss
g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a * 10.0 + cyc_loss_b * 10.0 + identity_loss

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
init_op = tf.global_variables_initializer()
sess.run(init_op)

#''' saver '''
saver = tf.train.Saver(max_to_keep=None)

###############################
# Restore facenet
###############################
def get_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    value_dict = {}
    for key in sorted(var_to_shape_map):
        value_dict[key] = reader.get_tensor(key)
    return value_dict

#value_dict = get_tensors_in_checkpoint_file('identity_gan/20180402-114759/model-20180402-114759.ckpt-275')
value_dict = get_tensors_in_checkpoint_file('identity_gan/20180408-102900/model-20180408-102900.ckpt-90')

#for var in tf.trainable_variables():
#    print(var.name)

facenet_variables = [var for var in tf.trainable_variables() if var.name.find('facenet') > -1]

assign_ops = []
for var in facenet_variables:
    varname = var.name[8:-2]
    value = value_dict[varname]
    assign_ops.append(var.assign(value))
    #print(var.name)
sess.run(assign_ops)

#####################################
# train
#####################################
#'''train'''


losses = []
for i in range(0, 10001):
    a_real_batch, b_real_batch = batchgen.generate_batch(batch_size)
    a_real_batch, b_real_batch = (a_real_batch*2)-1, (b_real_batch*2)-1

    # train G a+b
    _, glossa, glossb, cyclossa, cyclossb, idloss = sess.run([g_train_op, g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b, identity_loss], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    losses.append((glossa, glossb, cyclossa, cyclossb, idloss))

    # train D a
    d_summary_a_opt = sess.run([d_a_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # train D b
    d_summary_b_opt = sess.run([d_b_train_op], feed_dict={a_real: a_real_batch, b_real: b_real_batch})

    # save sample
    if i % 100 == 0:
        a2b_output, a2b2a_output, b2a_output, b2a2b_output = sess.run([a2b, a2b2a, b2a, b2a2b],
                                                                      feed_dict={a_real: a_real_batch[0:1],
                                                                                 b_real: b_real_batch[0:1]})
        sample = np.concatenate(
            [a_real_batch[0:1], a2b_output, a2b2a_output, b_real_batch[0:1], b2a_output,
             b2a2b_output], axis=1)
        sample = sample.reshape(crop_size * 6, crop_size, 3)
        save_path = 'identity_gan/samples/sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 1000 == 0:
        save_path = saver.save(sess, 'identity_gan/saved_models/nn_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))

genlosses_a = [x[0] for x in losses]
genlosses_b = [x[1] for x in losses]

cyc_losses_a = [x[2] for x in losses]
cyc_losses_b = [x[3] for x in losses]

id_losses = [x[4] for x in losses]


plt.plot(genlosses_a)
plt.plot(genlosses_b)
plt.plot(cyc_losses_a)
plt.plot(cyc_losses_b)
plt.plot(id_losses)
plt.legend(['genloss a', 'genloss b', 'cycloss a', 'cyclosss b', 'idloss'])
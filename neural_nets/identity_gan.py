from functools import partial
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import importlib

import neural_nets.nn as nn


class IdentityGan:

    def __init__(self,
                 crop_size=100,
                 lr=.0001):

        ##################
        # session
        ##################
        sesssion_config = tf.ConfigProto(allow_soft_placement=True)
        sesssion_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sesssion_config)

        ##############################
        # Facenet for identity vector
        ##############################
        network = importlib.import_module('neural_nets.resnet_model.inception_resnet_v1')

        def get_identity_vector(img, reuse):
            with tf.variable_scope('facenet', reuse=reuse):
                prelogits, endpoints = network.inference(images=img, keep_probability=1, phase_train=False,
                                                    bottleneck_layer_size=512, weight_decay=0.0)


                embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                return embeddings


        #""" graph """
        # resnet_model
        self.generator_a2b = partial(nn.generator, scope='a2b')
        self.generator_b2a = partial(nn.generator, scope='b2a')
        self.discriminator_a = partial(nn.discriminator, scope='a')
        self.discriminator_b = partial(nn.discriminator, scope='b')

        # Placeholders
        self.a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

        # Generator outputs
        self.a2b = self.generator_a2b(self.a_real)
        self.a2b2a = self.generator_b2a(self.a2b)

        self.b2a = self.generator_b2a(self.b_real)
        self.b2a2b = self.generator_a2b(self.b2a)

        # Discriminator outputs
        # a2b
        self.a_logit = self.discriminator_a(self.a_real)
        self.b2a_logit = self.discriminator_a(self.b2a)

        # b2a
        self.b_logit = self.discriminator_b(self.b_real)
        self.a2b_logit = self.discriminator_b(self.a2b)

        # Generator losses
        # Domain loss
        self.g_loss_a2b = tf.losses.mean_squared_error(self.a2b_logit, tf.ones_like(self.a2b_logit))
        self.g_loss_b2a = tf.losses.mean_squared_error(self.b2a_logit, tf.ones_like(self.b2a_logit))

        # Cycle loss
        self.cyc_loss_a = tf.losses.absolute_difference(self.a_real, self.a2b2a)
        self.cyc_loss_b = tf.losses.absolute_difference(self.b_real, self.b2a2b)

        # Identity loss
        self.identity_a_before = get_identity_vector(self.a_real, reuse=False)
        self.identity_a_after = get_identity_vector(self.a2b, reuse=True)
        self.identity_diff_a = tf.norm(self.identity_a_before - self.identity_a_after, axis=1)

        self.identity_b_before = get_identity_vector(self.b_real, reuse=True)
        self.identity_b_after = get_identity_vector(self.b2a, reuse=True)
        self.identity_diff_b = tf.norm(self.identity_b_before - self.identity_b_after, axis=1)

        self.identity_loss = tf.reduce_mean(self.identity_diff_a + self.identity_diff_b)

        # Sum loss
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a + self.cyc_loss_a * 10.0 + self.cyc_loss_b * 10.0 + self.identity_loss

        # Discriminator losses
        # Discriminator a losses
        self.d_loss_a_real = tf.losses.mean_squared_error(self.a_logit, tf.ones_like(self.a_logit))
        self.d_loss_b2a = tf.losses.mean_squared_error(self.b2a_logit, tf.zeros_like(self.b2a_logit))
        self.d_loss_a = self.d_loss_a_real + self.d_loss_b2a

        # Discriminator b losses
        self.d_loss_b_real = tf.losses.mean_squared_error(self.b_logit, tf.ones_like(self.b_logit))
        self.d_loss_a2b = tf.losses.mean_squared_error(self.a2b_logit, tf.zeros_like(self.a2b_logit))
        self.d_loss_b = self.d_loss_b_real + self.d_loss_a2b

        # Optimization
        t_var = tf.trainable_variables()
        d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
        d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
        g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

        self.d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.d_loss_a, var_list=d_a_var)
        self.d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.d_loss_b, var_list=d_b_var)
        self.g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.g_loss, var_list=g_var)

        # """ train """
        # ''' init '''
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

        #''' saver '''
        self.saver = tf.train.Saver(max_to_keep=None)

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

        #value_dict = get_tensors_in_checkpoint_file('resnet_model/20180402-114759/model-20180402-114759.ckpt-275')
        value_dict = get_tensors_in_checkpoint_file('resnet_model/20180408-102900/model-20180408-102900.ckpt-90')

        #for var in tf.trainable_variables():
        #    print(var.name)

        facenet_variables = [var for var in tf.trainable_variables() if var.name.find('facenet') > -1]

        assign_ops = []
        for var in facenet_variables:
            varname = var.name[8:-2]
            value = value_dict[varname]
            assign_ops.append(var.assign(value))
            #print(var.name)
        self.sess.run(assign_ops)
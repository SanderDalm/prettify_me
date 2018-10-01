from functools import partial
import tensorflow as tf
import neural_nets.nn as nn
from tensorflow.python import pywrap_tensorflow
import importlib


class IdentityGan:

    def __init__(self,
                 crop_size=100,
                 lr=.0001,
                 identity_weight=1):

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
        self.generator = partial(nn.generator, scope='generator')
        self.discriminator = partial(nn.discriminator, scope='discriminator')


        # Placeholders
        self.real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.input_face = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

        # Generator outputs
        self.generator_output = self.generator(self.input_face)

        # Discriminator outputs
        self.discriminator_output_real = self.discriminator(self.real)
        self.discriminator_output_fake = self.discriminator(self.generator_output)

        # Identity loss
        self.identity_before = get_identity_vector(self.input_face, reuse=False)
        self.identity_after = get_identity_vector(self.generator_output, reuse=True)
        self.identity_loss = tf.reduce_mean(tf.norm(self.identity_before - self.identity_after, axis=1))

        # Generator loss
        self.g_loss_without_identity = tf.losses.mean_squared_error(self.discriminator_output_fake, tf.ones_like(self.discriminator_output_fake))
        self.g_loss = self.g_loss_without_identity + self.identity_loss*identity_weight

        # Discriminator loss
        self.d_loss_real = tf.losses.mean_squared_error(self.discriminator_output_real, tf.ones_like(self.discriminator_output_real))
        self.d_loss_fake = tf.losses.mean_squared_error(self.discriminator_output_fake, tf.zeros_like(self.discriminator_output_fake))
        self.d_loss = self.d_loss_real + self.d_loss_fake


        # Optimization
        t_var = tf.trainable_variables()
        d_var = [var for var in t_var if 'discriminator' in var.name]
        g_var = [var for var in t_var if 'generator' in var.name]

        self.d_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.d_loss, var_list=d_var)
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

        # value_dict = get_tensors_in_checkpoint_file('resnet_model/20180402-114759/model-20180402-114759.ckpt-275')
        value_dict = get_tensors_in_checkpoint_file(
            'neural_nets/resnet_model/20180408-102900/model-20180408-102900.ckpt-90')

        # for var in tf.trainable_variables():
        #    print(var.name)

        facenet_variables = [var for var in tf.trainable_variables() if var.name.find('facenet') > -1]

        assign_ops = []
        for var in facenet_variables:
            varname = var.name[8:-2]
            value = value_dict[varname]
            assign_ops.append(var.assign(value))
            # print(var.name)
        self.sess.run(assign_ops)
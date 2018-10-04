from functools import partial
import tensorflow as tf
import neural_nets.nn as nn


class CycleGan:

    def __init__(self,
                 crop_size=100,
                 lr=.0001,
                 cycle_weight=1):

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

        # Sum loss
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a + self.cyc_loss_a * cycle_weight + self.cyc_loss_b * cycle_weight

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
        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

        #''' saver '''
        self.saver = tf.train.Saver(max_to_keep=None)
from functools import partial
import tensorflow as tf
import neural_nets.nn as nn

class SimpleGan:

    def __init__(self,
                 crop_size=100,
                 lr=.0001):

        #""" graph """
        # resnet_model
        self.generator = partial(nn.generator, scope='generator')
        self.discriminator = partial(nn.discriminator, scope='discriminator')


        # Placeholders
        self.real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.noise = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

        # Generator outputs
        self.generator_output = self.generator(self.noise)

        # Discriminator outputs
        self.discriminator_output_real = self.discriminator(self.real)
        self.discriminator_output_fake = self.discriminator(self.generator_output)

        # Generator loss
        self.g_loss = tf.losses.sigmoid_cross_entropy(logits=self.discriminator_output_fake, multi_class_labels=tf.ones_like(self.discriminator_output_fake))

        # Discriminator loss
        self.d_loss_real = tf.losses.sigmoid_cross_entropy(logits=self.discriminator_output_real, multi_class_labels=tf.ones_like(self.discriminator_output_real))
        self.d_loss_fake = tf.losses.sigmoid_cross_entropy(logits=self.discriminator_output_fake, multi_class_labels=tf.zeros_like(self.discriminator_output_fake))
        self.d_loss = self.d_loss_real + self.d_loss_fake


        # Optimization
        t_var = tf.trainable_variables()
        d_var = [var for var in t_var if 'discriminator' in var.name]
        g_var = [var for var in t_var if 'generator' in var.name]

        self.d_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.d_loss, var_list=d_var)
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
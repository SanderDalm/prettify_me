import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from age_regressor.batch_generator import BatchGenerator
import config


def Conv2D(x, filters, kernel_size, stride, padding='same'):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=stride,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.keras.initializers.he_normal(),
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                            padding=padding)


def UpConv2D(x, filters, kernel_size, stride, padding='same'):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      strides=stride,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.keras.initializers.he_normal(),
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                      activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                                      padding=padding)


def sample_noise(batch_size, dim=[25, 25, 32]):
    return np.random.uniform(0., 1., size=[batch_size]+dim)


def generator(noise, reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        x = UpConv2D(x=noise, filters=16, kernel_size=3, stride=1)
        x = UpConv2D(x=x, filters=16, kernel_size=3, stride=2)
        x = UpConv2D(x=x, filters=16, kernel_size=3, stride=2)
        x = UpConv2D(x=x, filters=3, kernel_size=3, stride=2)
    return tf.clip_by_value(x, 0, 1)


def discriminator(img, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):

        filter_size = 4
        x = Conv2D(img, filters=filter_size, kernel_size=3, stride=1)
        x = Conv2D(x, filters=filter_size, kernel_size=3, stride=1)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = Conv2D(x, filters=filter_size*2, kernel_size=3, stride=1)
        x = Conv2D(x, filters=filter_size*2, kernel_size=3, stride=1)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = Conv2D(x, filters=filter_size*4, kernel_size=3, stride=1)
        x = Conv2D(x, filters=filter_size*4, kernel_size=3, stride=1)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = Conv2D(x, filters=filter_size*8, kernel_size=3, stride=1)
        x = Conv2D(x, filters=filter_size*8, kernel_size=3, stride=1)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = Conv2D(x, filters=filter_size*16, kernel_size=3, stride=1)
        x = Conv2D(x, filters=filter_size*16, kernel_size=3, stride=1)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = tf.layers.flatten(x)

        x = tf.layers.dense(inputs=x, units=1, activation=None)

    return x

H = 32
W = 32
NOISE_DIM1, NOISE_DIM2, NOISE_DIM3 = int(H/8), int(W/8), 1
bg = BatchGenerator(path=config.datadir, height=H, width=W, datasets=['utkf'])
x, y = bg.generate_train_batch(32)
#plt.imshow(x[0])

# Define graph
real_img = tf.placeholder(tf.float32, [None, H, W, 3])
noise = tf.placeholder(tf.float32, [None, NOISE_DIM1, NOISE_DIM2, NOISE_DIM3])

fake_img = generator(noise)
r_logits = discriminator(real_img)
f_logits = discriminator(fake_img, reuse=True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

gen_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=.5, epsilon=.1).minimize(gen_loss, var_list=gen_vars)  # G Train step
disc_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=.5, epsilon=.1).minimize(disc_loss, var_list=disc_vars)  # D Train step

sess = tf.Session(config=tf.ConfigProto())#log_device_placement=True
tf.global_variables_initializer().run(session=sess)

batch_size = 32
nd_steps = 10
ng_steps = 10


d_losses = []
g_losses = []

for i in range(1000):
    print(i)

    for _ in range(nd_steps):
        img_batch, _ = bg.generate_train_batch(batch_size)
        noise_batch = sample_noise(batch_size, [NOISE_DIM1, NOISE_DIM2, NOISE_DIM3])
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={real_img: img_batch, noise: noise_batch})
    d_losses.append(dloss)

    if i > 200:
        for _ in range(ng_steps):
            img_batch, _ = bg.generate_train_batch(batch_size)
            noise_batch = sample_noise(batch_size, [NOISE_DIM1, NOISE_DIM2, NOISE_DIM3])
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={noise: noise_batch})
        g_losses.append(gloss)


plt.plot(d_losses, alpha=.8, color='g')
plt.plot(g_losses, alpha=.8, color='b')

generated_image = sess.run([fake_img], feed_dict={noise: sample_noise(batch_size, [NOISE_DIM1, NOISE_DIM2, NOISE_DIM3])})
img = generated_image[0][0]
img
img.shape
np.min(img)
np.max(img)
plt.imshow(img)
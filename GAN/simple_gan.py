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

def sample_Z(batch_size, dim=[25, 25, 32]):
    return np.random.uniform(-1., 1., size=[batch_size]+dim)

def sample_data(batch_size):
    x, y = bg.generate_train_batch(batch_size)
    return x


def generator(Z, reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
            x = UpConv2D(x=Z, filters=16, kernel_size=3, stride=1)
            x = UpConv2D(x=x, filters=16, kernel_size=3, stride=2)
            x = UpConv2D(x=x, filters=16, kernel_size=3, stride=2)
            x = UpConv2D(x=x, filters=3, kernel_size=3, stride=2)
        return x



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

    return x


H = 200
W = 200
DIM = int(H/8)
bg = BatchGenerator(path=config.datadir, height=H, width=W, datasets=['utkf'])


###############

Zp = tf.placeholder(tf.float32, [None, DIM, DIM, 32])
test = generator(Zp, False)
test2 = discriminator(test)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)


Z = sample_Z(1, [DIM, DIM, 32])
Z.shape

test, test2 = sess.run([test, test2], feed_dict={Zp: Z})

test.shape
test2.shape
plt.imshow(test[0])

###############


X = tf.placeholder(tf.float32, [None, H, W, 3])
Z = tf.placeholder(tf.float32, [None, DIM])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 4
nd_steps = 10
ng_steps = 10

for i in range(10001):
    X_batch = sample_data(batch_size, 2)
    Z_batch = sample_Z(batch_size, 32, 32)

    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

print(sess.run([G_sample], feed_dict={Z: sample_Z(1, 2)}))
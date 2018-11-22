import numpy as np
from scipy.misc import imsave

from neural_nets.simple_gan import SimpleGan
from batch_generators.batch_generators import OneClassBatchGenerator
from batch_generators.batch_gen_utils import get_two_classes_celeba, get_anime


crop_size = 128
lr = .0001
batch_size = 16
wasserstein = True

########################
# Batch gen
########################

anime_faces = get_anime()

batchgen = OneClassBatchGenerator(file_list=anime_faces, height=crop_size, width=crop_size)

########################
# Simple Gan
########################

gan = SimpleGan(crop_size=crop_size,
                lr=lr,
                wasserstein=wasserstein)

i = 0
while True:

    i += 1

    imgs = batchgen.generate_batch(batch_size)
    imgs = (imgs * 2) - 1
    noise_batch = np.random.uniform(-1, 1, [batch_size*2, crop_size, crop_size, 3])

    # train G
    _ = gan.sess.run([gan.g_train_op], feed_dict={gan.noise: noise_batch})

    # train D
    _ = gan.sess.run([gan.d_train_op], feed_dict={gan.real: imgs, gan.noise: noise_batch})

    # save sample
    if i % 100 == 0:
        samples = gan.sess.run([gan.generator_output],
                               feed_dict={gan.noise: noise_batch[0:6]})
        sample = np.concatenate([samples], axis=0)

        sample = sample.reshape(crop_size * 6, crop_size, 3)
        save_path = 'samples/simple_gan_sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 10000 == 0:
        save_path = gan.saver.save(gan.sess, 'models/simple_gan_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))
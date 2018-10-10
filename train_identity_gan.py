import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

from batch_generators.two_class_batch_generator import TwoClassBatchGenerator
from batch_generators.batch_gen_utils import get_two_classes_celeba
from neural_nets.identity_gan import IdentityGan

crop_size = 200
lr = .0001
batch_size = 16

########################
# Batch gen
########################

neg, pos = get_two_classes_celeba(attr='young', HQ=False)

batchgen = TwoClassBatchGenerator(file_list_a=neg, file_list_b=pos, height=crop_size, width=crop_size)

# n, p = batchgen.generate_batch(12)
#
# n = np.concatenate([n[0:6]], axis=0)
# n = n.reshape([crop_size * 6, crop_size, 3])
# p = np.concatenate([p[0:6]], axis=0)
# p = p.reshape([crop_size * 6, crop_size, 3])
# t = np.concatenate([n, p], axis=1)
#
# plt.imshow(t)

########################
# Identity gan
########################

gan = IdentityGan(crop_size=crop_size,
                  identity_weight=1)

i = 0
while True:

    i += 1

    face_batch, composite_batch = batchgen.generate_batch(batch_size)
    face_batch, composite_batch = (face_batch * 2) - 1, (composite_batch * 2) - 1

    # train G
    _ = gan.sess.run([gan.g_train_op], feed_dict={gan.input_face: face_batch})

    # train D
    _ = gan.sess.run([gan.d_train_op], feed_dict={gan.input_face: face_batch, gan.real: composite_batch})

    # save sample
    if i % 1000 == 0:
        output = gan.sess.run([gan.generator_output],
                               feed_dict={gan.input_face: face_batch[0:1]})
        sample = np.concatenate([face_batch[0], output[0][0]], axis=0)

        sample = sample.reshape(crop_size * 2, crop_size, 3)
        save_path = 'samples/identity_gan_sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 10000 == 0:
        save_path = gan.saver.save(gan.sess, 'models/identity_gan_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))
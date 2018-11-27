import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

from batch_generators.batch_generators import TwoClassBatchGenerator
from batch_generators.batch_gen_utils import get_two_classes_celeba, get_anime, get_women_UTKFACE
from neural_nets.cycle_gan import CycleGan


crop_size = 100
lr = .0001
batch_size = 16
wasserstein = False

########################
# Batch gen
########################

women = get_women_UTKFACE()

anime = get_anime()

batchgen = TwoClassBatchGenerator(file_list_a=women, file_list_b=anime, height=crop_size, width=crop_size)

n, p = batchgen.generate_batch(12)

n = np.concatenate([n[0:6]], axis=0)
n = n.reshape([crop_size * 6, crop_size, 3])
p = np.concatenate([p[0:6]], axis=0)
p = p.reshape([crop_size * 6, crop_size, 3])
t = np.concatenate([n, p], axis=1)

plt.imshow(t)


########################
# Cycle Gan
########################

gan = CycleGan(cycle_weight=10,
               crop_size=crop_size,
               wasserstein=wasserstein)

i = 0
while True:
    i += 1

    a_real_batch, b_real_batch = batchgen.generate_batch(batch_size)
    a_real_batch, b_real_batch = (a_real_batch * 2) - 1, (b_real_batch * 2) - 1

    # train G a+b
    g_summary_opt = gan.sess.run([gan.g_train_op], feed_dict={gan.a_real: a_real_batch, gan.b_real: b_real_batch})

    # train D a
    d_summary_a_opt = gan.sess.run([gan.d_a_train_op], feed_dict={gan.a_real: a_real_batch, gan.b_real: b_real_batch})

    # train D b
    d_summary_b_opt = gan.sess.run([gan.d_b_train_op], feed_dict={gan.a_real: a_real_batch, gan.b_real: b_real_batch})

    # save sample
    if i % 100 == 0:
        a2b_output, a2b2a_output, b2a_output, b2a2b_output = gan.sess.run([gan.a2b, gan.a2b2a, gan.b2a, gan.b2a2b],
                                                                      feed_dict={gan.a_real: a_real_batch[0:1],
                                                                                 gan.b_real: b_real_batch[0:1]})
        sample = np.concatenate(
            [a_real_batch[0:1], a2b_output, a2b2a_output, b_real_batch[0:1], b2a_output,
             b2a2b_output], axis=0)
        sample = sample.reshape(crop_size * 6, crop_size, 3)
        save_path = 'samples/cycle_gan_sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 10000 == 0:
        save_path = gan.saver.save(gan.sess, 'models/cycle_gan_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))
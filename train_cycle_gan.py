import numpy as np
from scipy.misc import imsave

from batch_generators.two_class_batch_generator import TwoClassBatchGenerator
from batch_generators.batch_gen_utils import get_two_classes_celeba
from neural_nets.cycle_gan import CycleGan


crop_size = 128
lr = .0001
batch_size = 16

########################
# Batch gen
########################

neg, pos = get_two_classes_celeba(attr='young', HQ=False)

batchgen = TwoClassBatchGenerator(file_list_a=neg, file_list_b=pos, height=crop_size, width=crop_size)

########################
# Cycle Gan
########################

gan = CycleGan(cycle_weight=10,
               crop_size=crop_size)

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
    if i % 1000 == 0:
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
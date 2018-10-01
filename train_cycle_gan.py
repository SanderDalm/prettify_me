import numpy as np
from scipy.misc import imsave
from glob import glob

from batch_generators.two_class_batch_generator import TwoClassBatchGenerator
from neural_nets.cycle_gan import CycleGan
import config


crop_size = 100
lr = .0001
batch_size = 16

########################
# Batch gen
########################
file_list_a = glob(config.datadir+'/UTKFace/*')
mislabeled = []
for x in file_list_a:
    try:
        age = x.split('_')[-4].split('/')[-1]
        int(age)
    except:
        mislabeled.append(x)

file_list_a = [x for x in file_list_a if x not in mislabeled]


file_list_a = [x for x in file_list_a if 18 < int(x.split('_')[-4].split('/')[-1]) < 40] # young
file_list_a = [x for x in file_list_a if x.split('_')[-2]=='2']  # asian
file_list_a = [x for x in file_list_a if x.split('_')[-3]=='1'] # women


file_list_b = glob(config.datadir+'/composites/*')

batchgen = TwoClassBatchGenerator(file_list_a=file_list_a, file_list_b=file_list_b, height=crop_size, width=crop_size)

########################
# Cycle Gan
########################

gan = CycleGan()

#'''train'''
for i in range(1, 100001):

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
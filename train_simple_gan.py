import numpy as np
from scipy.misc import imsave
from glob import glob

from neural_nets.simple_gan import SimpleGan
from batch_generators.two_class_batch_generator import TwoClassBatchGenerator
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


# file_list_a = [x for x in file_list_a if 18 < int(x.split('_')[-4].split('/')[-1]) < 40] # young
# file_list_a = [x for x in file_list_a if x.split('_')[-2]=='2']  # asian
# file_list_a = [x for x in file_list_a if x.split('_')[-3]=='1'] # women

#file_list_a = glob(config.datadir+'/composites/*')
#file_list_b = glob(config.datadir+'/composites/*')
file_list_b = file_list_a

batchgen = TwoClassBatchGenerator(file_list_a=file_list_a, file_list_b=file_list_b, height=crop_size, width=crop_size)

########################
# Cycle Gan
########################

gan = SimpleGan()

#'''train'''
for i in range(1, 100001):

    a_real_batch, b_real_batch = batchgen.generate_batch(batch_size)
    a_real_batch, b_real_batch = (a_real_batch * 2) - 1, (b_real_batch * 2) - 1
    real = np.concatenate([a_real_batch, b_real_batch], axis=0)
    noise_batch = np.random.normal(0, 1, [batch_size*2, crop_size, crop_size, 3])

    # train G
    _ = gan.sess.run([gan.g_train_op], feed_dict={gan.noise: noise_batch})

    # train D
    _ = gan.sess.run([gan.d_train_op], feed_dict={gan.real: real, gan.noise: noise_batch})

    # save sample
    if i % 1000 == 0:
        samples = gan.sess.run([gan.generator_output],
                               feed_dict={gan.noise: noise_batch[0:6]})
        sample = np.concatenate([samples], axis=1)

        sample = sample.reshape(crop_size * 6, crop_size, 3)
        save_path = 'samples/simple_gan_sample_{}.jpg'.format(i)
        imsave(save_path, sample)
        print('Sample saved to {}.'.format(save_path))

    # save model
    if i % 1000 == 0:
        save_path = gan.saver.save(gan.sess, 'models/simple_gan_{}.ckpt'.format(i))
        print('Model saved to {}.'.format(save_path))
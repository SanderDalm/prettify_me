import numpy as np
from scipy.misc import imsave
from glob import glob
from neural_nets.identity_gan import IdentityGan
import matplotlib.pyplot as plt

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


file_list_a = [x for x in file_list_a if 18 < int(x.split('_')[-4].split('/')[-1]) < 40] # young
file_list_a = [x for x in file_list_a if x.split('_')[-2]=='2']  # asian
file_list_a = [x for x in file_list_a if x.split('_')[-3]=='1'] # women

file_list_b = glob(config.datadir+'/composites/*')

batchgen = TwoClassBatchGenerator(file_list_a=file_list_a, file_list_b=file_list_b, height=crop_size, width=crop_size)

########################
# Cycle Gan
########################

gan = IdentityGan()

#'''train'''
for i in range(1, 100001):

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


# face_batch, composite_batch = batchgen.generate_batch(batch_size)
# face_batch, composite_batch = (face_batch * 2) - 1, (composite_batch * 2) - 1
# id_loss, g_loss = gan.sess.run([gan.identity_loss, gan.g_loss_without_identity], feed_dict={gan.input_face: face_batch})
# id_loss
# g_loss

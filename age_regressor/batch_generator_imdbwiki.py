import os
from os.path import join
from skimage.color import gray2rgb
from scipy.misc import imread, imresize
import numpy as np
from glob import glob

from scipy.io import loadmat
from datetime import datetime


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


class BatchGenerator:

    def __init__(self, path=None, height=400, width=275, n_train=20000, datasets=[]):

        self.path = path
        self.height = height
        self.width = width

        self.images, self.labels = self.parse_data(datasets)
        shuffled_indices = list(range(len(self.images)))
        np.random.shuffle(shuffled_indices)
        self.images, self.labels = self.images[shuffled_indices], self.labels[shuffled_indices]
        self.images_train, self.labels_train = self.images[:n_train], self.labels[:n_train]
        self.images_val, self.labels_val = self.images[n_train:], self.labels[n_train:]


    def parse_imdbwiki_data(self, db):

        full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
            join(self.path, '{}_crop'.format(db), '{}.mat'.format(db)), db)

        filenames = []
        labels = []
        for index, img_path in enumerate(full_path):

            if face_score[index] < 1:
                continue

            if (~np.isnan(second_face_score[index])) and second_face_score[index] > 0.0:
                continue

            if not (0 <= age[index] <= 100):
                continue

            if np.isnan(gender[index]):
                continue

            else:
                filenames.append(join(self.path, '{}_crop'.format(db), full_path[index][0]))
                labels.append(age[index])

        return filenames, labels


    def parse_utkf_data(self):

        images = glob(join(self.path, 'UTKFace') + '/*')
        labels = []
        for path in images:
            label = path.split('/')[-1].split('_')[0]
            labels.append(int(label))
        return images, labels



    def parse_data(self, datasets):

        all_images = []
        all_labels = []

        if 'wiki' in datasets:
            images, labels = self.parse_imdbwiki_data('wiki')
            all_images.extend(images)
            all_labels.extend(labels)
        if 'imdb' in datasets:
            images, labels = self.parse_imdbwiki_data('imdb')
            all_images.extend(images)
            all_labels.extend(labels)
        if 'utkf' in datasets:
            images, labels = self.parse_utkf_data()
            all_images.extend(images)
            all_labels.extend(labels)

        return np.array(images), np.array(labels)


    def generate_batch(self, batch_size, images, labels):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            index = np.random.choice(range(len(images)))
            image = imread(images[index])
            if image.shape[0] < 64 or image.shape[1] < 64:
                continue
            if len(image.shape) == 2:
                image = gray2rgb(image)
            if image.shape[0] != self.height or image.shape[1] != self.width:
                image = imresize(image, [self.height, self.width, 3])
            image = image / 255

            x_batch.append(image)
            y_batch.append(labels[index])

        return np.array(x_batch), np.array(y_batch).reshape(len(y_batch), 1)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_val, self.labels_val)


# bg = BatchGenerator(path='/home/sander/data/prettify_me', height=256, width=256, datasets=['utkf'])
#
# x, y = bg.generate_train_batch(16)
#
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.5, wspace=.001)
# axs = axs.ravel()
#
# for index, (img, label) in enumerate(zip(x, y)):
#
#     print(x.shape)
#
#     axs[index].imshow(img.reshape(256,256, 3))
#     axs[index].set_title('Labeled age: {}'.format(label))
#
# plt.show()
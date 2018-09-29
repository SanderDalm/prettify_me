import os
from os.path import join
from collections import defaultdict

from skimage.color import gray2rgb
from scipy.misc import imread, imresize
import numpy as np
from glob import glob

class AttributeBatchGenerator:

    def __init__(self, path=None, height=218, width=178, n_train=180000):

        self.path = path
        self.height = height
        self.width = width

        self.images, self.labels = self.parse_data()
        shuffled_indices = list(range(len(self.images)))
        np.random.shuffle(shuffled_indices)
        self.images, self.labels = self.images[shuffled_indices], self.labels[shuffled_indices]
        self.images_train, self.labels_train = self.images[:n_train], self.labels[:n_train]
        self.images_val, self.labels_val = self.images[n_train:], self.labels[n_train:]


    def parse_data(self):

        file_list = glob(self.path+'/img_align_celeba_png/*')
        file_list = sorted(file_list)

        label_dict = defaultdict(np.array)
        attribute_file = self.path+'/Anno/list_attr_celeba.txt'

        with open(attribute_file, 'r') as f:
            for index, line in enumerate(f.read().split('\n')):

                line_split = line.split(' ')
                line_split = [x.strip(' ') for x in line_split]
                line_split = [x for x in line_split if len(x) > 0]

                if len(line_split) > 0:

                    if index == 0:
                        continue
                    if index == 1:
                        self.attributes = line_split
                    else:
                        filename = line_split[0]
                        filename = filename.replace('jpg', 'png')  # fix apparent error in file
                        label_vector = line_split[1:]
                        label_vector = [0 if x == '-1' else 1 for x in label_vector]
                        label_dict[filename] = np.array(label_vector)

        keys = [x.split('/')[-1] for x in file_list]
        labels = [label_dict[key] for key in keys]

        return np.array(file_list), np.array(labels)


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

        return np.array(x_batch), np.array(y_batch).reshape(len(y_batch), len(self.attributes))


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_val, self.labels_val)


#bg = BatchGenerator(path='/home/sander/data/prettify_me')
#x, y = bg.generate_train_batch(16)
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
#     axs[index].imshow(img.reshape(218, 178, 3))
#     #axs[index].set_title('Labeled age: {}'.format(label))
#
# plt.show()
from os.path import join
from skimage.color import gray2rgb
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class TwoClassBatchGenerator:

    def __init__(self, file_list_a, file_list_b, height=200, width=200):

        self.file_list_a = file_list_a
        self.file_list_b = file_list_b
        self.height = height
        self.width = width

    def read_img(self, path):

        image = imread(path)
        if len(image.shape) == 2:  # black and white to rgb
            image = gray2rgb(image)
        if image.shape[2] == 4:    # remove alpha channel if needed
            image = image[:, :, :3]
        if image.shape[0] != self.height or image.shape[1] != self.width: # resize if needed
            image = imresize(image, [self.height, self.width, 3])


        image = image / 255
        return image

    def generate_batch(self, batch_size):

        imgs_a = []
        imgs_b = []

        for sample in range(batch_size):

                index = np.random.choice(range(len(self.file_list_a)))
                path = self.file_list_a[index]
                image = self.read_img(path)
                imgs_a.append(image)

                index = np.random.choice(range(len(self.file_list_b)))
                path = self.file_list_b[index]
                image = self.read_img(path)
                imgs_b.append(image)

        return np.array(imgs_a), np.array(imgs_b)



class OneClassBatchGenerator(TwoClassBatchGenerator):

    def __init__(self, file_list, height=200, width=200):

        self.file_list = file_list
        self.height = height
        self.width = width


    def generate_batch(self, batch_size):

        imgs = []
        for sample in range(batch_size):

                index = np.random.choice(range(len(self.file_list)))
                path = self.file_list[index]
                image = self.read_img(path)
                imgs.append(image)

        return np.array(imgs)



#
# import config
#
# file_list = glob(config.datadir+'/anime/*/*.png')
#
# bg = OneClassBatchGenerator(file_list=file_list, height=200, width=200)
#
# imgs = bg.generate_batch(16)
#
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.5, wspace=.001)
# axs = axs.ravel()
#
# for index, img in enumerate(imgs):
#     axs[index].imshow(img.reshape(200,200, 3))
# plt.show()
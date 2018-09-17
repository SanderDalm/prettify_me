from os.path import join
from skimage.color import gray2rgb
from scipy.misc import imread, imresize
import numpy as np
from glob import glob

class Men_Women_BatchGenerator:

    def __init__(self, path=None, height=200, width=200):

        self.path = path
        self.height = height
        self.width = width
        self.men, self.women = self.parse_data()

    def parse_data(self):

        images = glob(join(self.path, 'UTKFace') + '/*')
        men = []
        women = []
        for path in images:
            sex = int(path.split('/')[-1].split('_')[1])
            if sex == 0:
                men.append(path)
            else:
                women.append(path)
        return men, women

    def read_img(self, path):

        image = imread(path)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = imresize(image, [self.height, self.width, 3])
        image = image / 255
        return image

    def generate_batch(self, batch_size):

        men = []
        women = []

        for sample in range(batch_size):

                index = np.random.choice(range(len(self.men)))
                path = self.men[index]
                image = self.read_img(path)
                men.append(image)

                index = np.random.choice(range(len(self.women)))
                path = self.women[index]
                image = self.read_img(path)
                women.append(image)

        return np.array(men), np.array(women)

# import config
# bg = Men_Women_BatchGenerator(path=config.datadir, height=200, width=200)
#
# men, women = bg.generate_batch(16)
#
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.5, wspace=.001)
# axs = axs.ravel()
#
# for index, img in enumerate(women):
#     axs[index].imshow(img.reshape(200,200, 3))
# plt.show()
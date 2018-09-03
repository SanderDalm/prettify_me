import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.color import gray2rgb
from scipy.misc import imread, imresize
from age_regressor.batch_generator import BatchGenerator
from age_regressor.age_regressor import AgeRegressor

########################################
# Set globals
########################################0

HEIGHT = 200
WIDTH = 200
BATCH_SIZE = 32
NUM_STEPS = 5001
DROPOUT = .5
AUGMENT = 1
DECAY = 1

bg = BatchGenerator(path='/mnt/ssd/data/prettify_me', height=WIDTH, width=WIDTH, n_train=20000, datasets=['utkf'])

#x, y = bg.generate_train_batch(1)
#len(bg.images_train)
#len(bg.images_val)


nn = AgeRegressor(HEIGHT, WIDTH)
#nn.load_weights('models/neural_net5000.ckpt')

loss, val_loss = nn.train(num_steps=NUM_STEPS,
                          batchgen=bg,
                          batch_size=BATCH_SIZE,
                          dropout_rate=DROPOUT,
                          augment=AUGMENT,
                          lr=.0001,
                          decay=DECAY)

plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()


####################################
# Check prediction quality
####################################

preds = []
labels = []
for filename in bg.images_val:

    image = imread(filename)
    if image.shape[0] < 64 or image.shape[1] < 64:
        continue
    if len(image.shape) == 2:
        image = gray2rgb(image)
    if image.shape[0] != HEIGHT or image.shape[1] != WIDTH:
        image = imresize(image, [HEIGHT, WIDTH, 3])
    image = image / 255

    pred = nn.predict(image)[0]
    preds.append(pred)

    label = filename.split('/')[-1].split('_')[0]
    labels.append(int(label))

plt.scatter(labels, preds, alpha=.8)
plt.show()
print(pearsonr(labels, preds)[0])


x, y = bg.generate_val_batch(16)

fig, axs = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()

for index, (img, label) in enumerate(zip(x, y)):
    pred = nn.predict(img)

    axs[index].imshow(img.reshape(HEIGHT, WIDTH, 3))
    axs[index].set_title('Labeled age: {} Pred: {}'.format(label, pred))
plt.savefig('test')


########################################
# Visualize convolutional layers
########################################

import tensorflow as tf
import numpy as np
import cv2

def visualize_layers(nn, layer_number, input_img):

    def visualize_layer(nn, op, input, n_iter, stepsize):
        t_score = tf.reduce_mean(op)
        t_grad = tf.gradients(t_score, nn.x)[0]

        img = input.copy()
        for i in range(n_iter):
            g, score = nn.session.run([t_grad, t_score], {nn.x: img, nn.augment: 0, nn.dropout_rate: 0})
            g /= g.std() + 1e-8
            img += g * stepsize
            img -= np.mean(img)
            img /= np.std(img)
            img += abs(np.min(img))
            img /= np.max(img)
            if i%5==0:
                img = cv2.blur(img.reshape(HEIGHT, WIDTH, 3), (5, 5)).reshape(1, HEIGHT, WIDTH, 3)
            #print(score, end=' ')
        if len(img.shape) == 3:
            return img.reshape(nn.height, nn.width)
        if len(img.shape) == 4:
            return img.reshape(nn.height, nn.width, 3)

    layers = [op for op in nn.session.graph.get_operations() if op.type == 'Conv2D']
    layer = layers[layer_number]
    layername = layer.name
    print(layername)
    target = nn.session.graph.get_tensor_by_name(layername + ':0')
    num_channels = target.shape.as_list()[1]

    num_rows = 4#int(np.sqrt(num_channels)) + 1
    num_cols = 4#num_rows

    fig, axs = plt.subplots(num_rows, num_cols, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.001, wspace=.001)
    axs = axs.ravel()

    for channel in range(num_channels):
        print(channel)
        img = visualize_layer(nn=nn,
                              input=input_img.copy(),
                              op=target[:, :, :, channel],
                              n_iter=1000,
                              stepsize=.01)

        axs[channel].imshow(img)
    plt.show()

input_img = imread("/home/sander/sander/Foto's/2016/HyunJeong/20160701_153459.jpg")
input_img = imresize(input_img, [200, 200])
#plt.imshow(input_img)
#plt.show()
input_img = input_img.reshape(1, 200, 200, 3)
input_img = input_img.astype(np.float32)
input_img = np.random.normal(.5, 5, size=(1, HEIGHT, WIDTH, 3))
visualize_layers(nn=nn, layer_number=-1, input_img=input_img)

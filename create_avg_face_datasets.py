from composite_faces.face_average import average_faces, readImages
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.misc import imsave
import config

# UTKF dataset
h = 200
w = 200
image_paths = glob(config.datadir+'/UTKFace/*.jpg')

mislabeled = []

for x in image_paths:
    try:
        age = x.split('_')[-4].split('/')[-1]
        int(age)
    except:
        mislabeled.append(x)

image_paths = [x for x in image_paths if x not in mislabeled]

men = [x for x in image_paths if x.split('_')[-3]=='0']
women = [x for x in image_paths if x.split('_')[-3]=='1']

white = [x for x in image_paths if x.split('_')[-2]=='0']
black = [x for x in image_paths if x.split('_')[-2]=='1']
asian = [x for x in image_paths if x.split('_')[-2]=='2']
indian = [x for x in image_paths if x.split('_')[-2]=='3']

kids = [x for x in image_paths if int(x.split('_')[-4].split('/')[-1]) < 12]
teenagers = [x for x in image_paths if 12 < int(x.split('_')[-4].split('/')[-1]) < 20]
young_adults =  [x for x in image_paths if 18 < int(x.split('_')[-4].split('/')[-1]) < 40]
adults =  [x for x in image_paths if 29 < int(x.split('_')[-4].split('/')[-1]) < 60]
old =  [x for x in image_paths if 59 < int(x.split('_')[-4].split('/')[-1])]

selection = [x for x in image_paths if x in asian and x in women and x in young_adults]

landmarks = [[(52, 70), (120, 70)] for _ in range(len(selection))]
eye_pos = landmarks[0]

# selection.sort()
# orig_face = selection[23:24]
# plt.imshow(readImages(orig_face))
# selection = orig_face*200 + selection
#
# avg_face = average_faces(image_paths=selection,
#                              landmarks=landmarks,
#                              eyecornerDst=eye_pos,
#                              h=h,
#                              w=w)
#
# plt.imshow(avg_face)

def create_random_composite(selection, n):
    indices = np.random.choice(range(len(selection)), n)
    subselection = np.array(selection)[indices]
    landmarks = [[(52, 70), (120, 70)] for _ in range(len(subselection))]

    composite = average_faces(image_paths=subselection,
                             landmarks=landmarks,
                             eyecornerDst=eye_pos,
                             h=h,
                             w=w)

    return composite

for i in range(1000):
    composite = create_random_composite(selection, 50)
    imsave(config.datadir+'/composites/{}.bmp'.format(i), composite)

#Celeba dataset
# h = 218
# w = 178
# image_paths = glob('/home/sander/data/prettify_me/img_align_celeba_png/*.png')
# image_paths.sort()
# landmarks = readPoints()
# eye_pos = [(69, 111), (108, 111)]

# fig, axs = plt.subplots(3, 3, facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.001, wspace=.001)
# axs = axs.ravel()
#
# for i in range(9):
#     axs[i].imshow(create_random_composite(selection))
# plt.show()

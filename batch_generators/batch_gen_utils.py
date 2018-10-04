from os.path import join
from glob import glob
from collections import defaultdict

import numpy as np

import config


def get_composites():
    return glob(config.datadir+'/composites/*')


def get_old_young_UTKFACE():
    file_list = glob(join(config.datadir, '/UTKFace/*'))

    mislabeled = []
    for x in file_list:
        try:
            age = x.split('_')[-4].split('/')[-1]
            int(age)
        except:
            mislabeled.append(x)

    file_list = [x for x in file_list if x not in mislabeled]

    file_list_a = [x for x in file_list if 18 < int(x.split('_')[-4].split('/')[-1]) > 50]  # old
    file_list_b = [x for x in file_list if 18 < int(x.split('_')[-4].split('/')[-1]) < 35]  # young

    return file_list_a, file_list_b


def get_two_classes_celeba(attr='young'):

    if attr == 'young':
        trait_number = -1
    if attr == 'attractive':
        trait_number = 2

    attribute_file = join(config.datadir, 'Anno/list_attr_celeba.txt')

    negatives = []
    positives = []
    with open(attribute_file, 'r') as f:
        for index, line in enumerate(f.readlines()):

            line_split = line.split(' ')
            line_split = [x.strip(' ') for x in line_split]
            line_split = [x.strip('\n') for x in line_split]
            line_split = [x for x in line_split if len(x) > 0]

            if len(line_split) > 0:
                if index < 2:
                    continue
                else:
                    filename = line_split[0]
                    filename = filename.replace('png', 'jpg')  # fix apparent error in file
                    label_vector = line_split[1:]
                    if label_vector[trait_number] == '-1':
                        negatives.append(join(config.datadir, 'img_align_celeba', filename))
                    if label_vector[trait_number] == '1':
                        positives.append(join(config.datadir, 'img_align_celeba', filename))
    return negatives, positives
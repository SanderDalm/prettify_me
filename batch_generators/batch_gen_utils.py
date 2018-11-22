from os.path import join
from glob import glob
from collections import defaultdict

import numpy as np

import config


def get_composites():
    return glob(config.datadir+'/composites/*')

def get_anime():
    return glob(config.datadir + '/anime/*/*.png')


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

    # file_list_a = [x for x in file_list_a if 18 < int(x.split('_')[-4].split('/')[-1]) < 40] # young
    # file_list_a = [x for x in file_list_a if x.split('_')[-2]=='2']  # asian
    # file_list_a = [x for x in file_list_a if x.split('_')[-3]=='1'] # women

    return file_list_a, file_list_b


def get_two_classes_celeba(attr='young', HQ=False):

    celeba_dir = 'img_align_celeba'
    if HQ:
        celeba_dir = 'celeba_hq/images'
        celeba_hq_files = glob(join(config.datadir, celeba_dir)+'/*')
        celeba_hq_files = [x.split('/')[-1].strip('.bmp') for x in celeba_hq_files]
        celeba_hq_files = sorted(celeba_hq_files)
        len(celeba_hq_files)


    if attr == 'young':
        trait_number = -1
    elif attr == 'attractive':
        trait_number = 2
    else:
        raise AssertionError


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

                    # positive case
                    if label_vector[trait_number] == '-1':
                        if HQ:
                            if filename in celeba_hq_files:
                                negatives.append(join(config.datadir, celeba_dir, filename) + '.bmp')
                        else:
                            negatives.append(join(config.datadir, celeba_dir, filename))

                    # negative case
                    if label_vector[trait_number] == '1':
                        if HQ:
                            if filename in celeba_hq_files:
                                positives.append(join(config.datadir, celeba_dir, filename) + '.bmp')
                        else:
                            positives.append(join(config.datadir, celeba_dir, filename))
    return negatives, positives



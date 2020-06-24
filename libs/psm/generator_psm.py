# @Time    : 2019-05-16 21:02
# @Author  : WeiLong Wu
# @File    : generator_psm_no_flip_and_rot.py
# @Function: rewrite the function to fit txt file input

import numpy as np
from scipy import misc, io
import random
from random import random as rand
from random import shuffle


def gen_flip_and_rot_psm(cover_txt, stego_txt, thread_idx, n_threads):
    with open(cover_txt) as f:
        cover_list = f.readlines()
        cover_list = [a.strip() for a in cover_list]

    with open(stego_txt) as f:
        stego_list = f.readlines()
        stego_list = [a.strip() for a in stego_list]

    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego txt file is empty"
    assert nb_data != 0, "the cover txt file is empty"
    assert len(stego_list) == nb_data, "the cover txt file and " + \
                                       "the stego txt file don't " + \
                                       "have the same number of files " + \
                                       "respectively %d and %d" % (nb_data, + \
        len(stego_list))
    load_mat = cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='uint8')

    iterable = zip(cover_list, stego_list)
    while True:
        shuffle(iterable)
        for cover_path, stego_path in iterable:
            if load_mat:
                batch[0, :, :, 0] = io.loadmat(cover_path)['im']
                batch[1, :, :, 0] = io.loadmat(stego_path)['im']
            else:
                batch[0, :, :, 0] = misc.imread(cover_path)
                batch[1, :, :, 0] = misc.imread(stego_path)
            rot = random.randint(0, 3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]


def gen_train_or_valid_psm(thinet_cover_txt, thinet_stego_txt, thread_idx, n_threads):
    with open(thinet_cover_txt) as f:
        cover_list = f.readlines()
        cover_list = [a.strip() for a in cover_list]

    with open(thinet_stego_txt) as f:
        stego_list = f.readlines()
        stego_list = [a.strip() for a in stego_list]

    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego txt file is empty"
    assert nb_data != 0, "the cover txt file is empty"
    assert len(stego_list) == nb_data, "the cover txt file and " + \
                                       "the stego txt file don't " + \
                                       "have the same number of files " + \
                                       "respectively %d and %d" % (nb_data, \
                                                                   len(stego_list))
    load_mat = cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='uint8')
    img_shape = img.shape

    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            if load_mat:
                batch[0, :, :, 0] = io.loadmat(cover_path)['im']
                batch[1, :, :, 0] = io.loadmat(stego_path)['im']
            else:
                batch[0, :, :, 0] = misc.imread(cover_path)
                batch[1, :, :, 0] = misc.imread(stego_path)
            yield [batch, labels]

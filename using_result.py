# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from data_sample import gaussian_mixture, gaussian_mixture_B, single_gaussian
from util import plot_scatter


if __name__ == u'__main__':

    file_name = './model.dump'

    z_dim = 256    
    batch_size = 256
    y = -2.0
    begin_x, end_x = 0.0, 6.0
    grid_size = (end_x - begin_x)/batch_size
    inputs = np.asarray([[grid_size * _, y] for _ in range(batch_size)], dtype = np.float32)

    # make model
    print('-- make model --')
    model = Model(z_dim)
    model.set_model()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, file_name)
        outputs = model.get_disc_value(sess, inputs)
        print(outputs)

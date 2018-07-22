# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data1, data2, filename):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.hist(data1, normed = True, color = 'blue', alpha = 0.5, bins=50)
    ax.hist(data2, normed = True, color = 'red', alpha = 0.5, bins=50)
    ax.set_ylim(0,1.0)
    ax.set_xlim(0,5.0)    
    plt.savefig(filename)

if __name__ == '__main__':
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    mu, sigma = 70, 10    
    x2 = mu + sigma * np.random.randn(10000)
    plot_histgram(x, x2, 'hogehoge.png')

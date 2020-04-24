# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:21:37 2020

@author: Qasymjomart
"""
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

multiclass = np.array([[332, 12, 1, 1],
                       [18, 45, 0, 0],
                       [6, 0, 74, 0],
                       [1, 1, 0, 30]])

class_names = ['ductal carcinoma', 'lobular carcinoma', 'mucinous carcinoma', 'papillary carcinoma']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.show()
plt.savefig('mal_mat.png')
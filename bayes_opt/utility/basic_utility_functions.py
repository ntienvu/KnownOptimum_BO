# -*- coding: utf-8 -*-

#from bayes_opt import PradaBayOptFn

#from sklearn.gaussian_process import GaussianProcess
#from scipy.stats import norm
#import matplotlib as plt

import numpy as np
import random
import time

#from bayes_opt import PradaBayOptFn


def generate_random_points(bounds,size=1):
    x_max = [np.random.uniform(x[0], x[1], size=size) for x in bounds]
    x_max=np.asarray(x_max)
    x_max=x_max.T
    return x_max
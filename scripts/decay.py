# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:26:47 2019

@author: c1672922
"""

import numpy as np
import matplotlib.pyplot as plt


def inv_decay(decay_const, U):
    return -(np.log(U)) / decay_const

def decay(decay_const, time):
    decay_prob = 1 - np.exp(-decay_const * time)
    return decay_prob

def prob_decay(decay_const, time, dt):
    return np.exp(-decay_const*time) * decay_const*dt

half_life = 2200
decay_const = np.log(2)/half_life

plt.figure(1)
test = list()
for i in range(100000):
    U = np.random.rand()
    test.append(inv_decay(decay_const, U))
plt.hist(test, bins=1000)



plt.figure(2)
data = list()
dt = 1000/10000
for i in np.linspace(0, 10000, num=10000):
    data.append(prob_decay(decay_const, i, dt))
plt.hist(np.cumsum(data), bins=100)



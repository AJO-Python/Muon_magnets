#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:33:25 2019

@author: josh
"""
import numpy as np
import matplotlib.pyplot as plt

from Modules.muon import Muon
import Modules.functions as func

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

N = 1000
field = 1e-3
particles = [Muon() for _ in range(N)]
for p in particles:
    p.apply_field(field_dir=[1,1,0], random_phase=True)
    p.set_kubo_toyabe(0.5)
lifetimes = map(lambda p: p.lifetime, particles)
rads = map(lambda p: (p.total_rads % 2*np.pi)/2*np.pi, particles)
kt = map(lambda p: p.kt, particles)

information = np.array([*lifetimes, *rads, *kt])
information.view("i8,i8,i8").sort(order=["f1"], axis=0)

plt.figure()
plt.plot(informatio[0], information[1])
plt.xlim(0, 20e-6)
plt.show()

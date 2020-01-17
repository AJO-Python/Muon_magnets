#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:33:25 2019

@author: josh
"""
import numpy as np
import matplotlib.pyplot as plt

from Modules.muon import Muon
from Modules.positron import Positron
import Modules.functions as func

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


N = 1000
field = 1e-3
particles = [Muon() for _ in range(N)]
for p in particles:
    p.apply_field(field_dir=[1,1,0], random_phase=True)
    p.get_kubo_toyabe(0.5)
lifetimes = list(map(lambda p: p.lifetime, particles))
rads = list(map(lambda p: p.total_rads % 2*np.pi, particles))
kt = list(map(lambda p: p.kt, particles))
plt.figure()
plt.scatter(lifetimes, kt)
plt.xlim(0, 20e-6)
plt.show()

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

lifetimes = list(map(lambda p: p.lifetime, particles))
asym = list(map(lambda p: p.get_asym(1/3, p.get_larmor(field)), particles))


plt.figure()
plt.scatter(lifetimes, asym)
plt.xlim(0, 20e-6)
plt.show()
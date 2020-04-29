#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:54:43 2020

@author: joshowen121
"""

import numpy as np
import matplotlib.pyplot as plt

from modules.grid import Grid
from modules.island import Island

isle_1 = Island(orientation=0, location=[0e-6, 0e-6, 0.0], strength=8e-8, size=[1.6e-6, 700e-9])
isle_2 = Island(orientation=0, location=[2e-6, 0.0, 0.0], strength=8e-8, size=[1.6e-6, 700e-9])
X = np.linspace(-20e-6, 20e-6, 200)
Y = np.copy(X)
Z = np.copy(X)

Bx1, By1, Bz1 = isle_1.fancy_mag_field(r=np.meshgrid(X, Y, Z))
Bx2, By2, Bz2 = isle_2.fancy_mag_field(r=np.meshgrid(X, Y, Z))

Bx = Bx1 + Bx2
By = By1 + By2
Bz = Bz1 + Bz2

# Bx, By, Bz = Bx1, By1, Bz1
fig, ax = plt.subplots()

ax.streamplot(X, Y, Bx[:, :, -1], By[:, :, -1], density=2)
ax.set_xlim(min(X), max(X))
ax.set_ylim(min(Y), max(Y))
ax.add_artist(isle_1.get_outline())
ax.add_artist(isle_2.get_outline())
ax.ticklabel_format(axis="both", style="sci", scilimits=(-6, -6))
plt.show()

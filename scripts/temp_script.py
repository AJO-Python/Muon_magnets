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
from modules.muon import Muon
from modules.ensemble import Ensemble
import modules.functions as func

isle_1 = Island(orientation=0, location=[-10e-6, 0, 0], strength=2e-8, size=[1.6e-6, 700e-9])
isle_2 = Island(orientation=180, location=[10e-6, 0, 0], strength=2e-8)
grid = Grid.__new__(Grid)
grid.islands = [isle_1]  # , isle_2]

locs = np.linspace(-50e-6, 50e-6, 500, endpoint=True)
particles = Ensemble.__new__(Ensemble)
particles.loc = locs
particles.run_name = "test"
particles.muons = np.array([Muon(loc=np.array((0, 0, i))) for i in locs])
particles.calculate_fields(grid)
particles.load_fields()
particles.set_relaxations()

# particles.show_on_plot()

fig, axs = plt.subplots(4, 1)
for ax, (key, item) in zip(axs, particles.field_dict.items()):
    ax.plot(locs, item)
    ax.set_ylabel("Field strength")
    ax.set_xlabel(f"Distance ({key})")
    ax.set_title(f"{key}-field against {key} distance")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
    ax.grid()
plt.show()

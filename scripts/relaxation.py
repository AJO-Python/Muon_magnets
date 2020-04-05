#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import modules.functions as func
from modules.muon import Muon

np.random.seed(1)
N = 10000
particles = [Muon() for _ in range(N)]
particles[0].lifetime = Muon.TIME_SCALE[-1]

field_x = np.random.normal(0, 0.1e-3, N)
field_y = np.random.normal(0, 0.1e-3, N)
field_z = np.random.normal(0, 0.1e-3, N)
fields = np.array(list(zip(field_x, field_y, field_z)))

magnitudes = [func.get_mag(f) for f in fields]
field_dict = {"total": magnitudes, "x": field_x, "y": field_y, "z": field_z}

# Get each muons polarisation
relaxations = np.array([p.full_relaxation(fields[i], False) for i, p in enumerate(particles)])

# Normalise sum
overall = np.nansum(relaxations, axis=0) / N

# Setup subplots
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
ax1 = plt.subplot2grid((2, 3), (0, 2))
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (1, 1))
ax4 = plt.subplot2grid((2, 3), (1, 2))

field_axes = (ax1, ax2, ax3, ax4)
if N < 100:
    for i in range(N):
        ax0.plot(Muon.TIME_SCALE, relaxations[i], alpha=0.5, lw=0.5)
ax0.plot(Muon.TIME_SCALE, overall, lw=2, c="k")
ax0.set_xlim(Muon.TIME_SCALE[0], Muon.TIME_SCALE[-1])
ax0.grid()
ax0.set_title("Zero field relaxation function")

ax1.set_title("Magnitudes of overall field")

for sub_ax, field in zip(field_axes, field_dict.keys()):
    sub_ax.hist(field_dict[field], bins=100)
    sub_ax.set_title(f"Magnitudes of {field}")
    sub_ax.set_xlabel("Field strength (T)")
    sub_ax.set_ylabel("Frequency")
    sub_ax.grid()
    sub_ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
# Add legend
legend_handles = {Line2D([0], [0],
                         color="g", markerfacecolor="w",
                         label="Individual muons"),
                  Line2D([0], [0],
                         color="k", markerfacecolor="k",
                         label="Summed relaxation functions")}
ax0.legend(handles=legend_handles, loc="upper right")
ax0.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))
plt.tight_layout(pad=1)
plt.show()

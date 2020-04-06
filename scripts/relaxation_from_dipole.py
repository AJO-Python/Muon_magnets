#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import modules.functions as func
from modules.muon import Muon
from modules.dipole import Dipole
from modules.model_equations import static_GKT

np.random.seed(1)

N = 10000
run_name = "5x5_U_3"
particles = [Muon(location=np.random.normal(loc=(6e-6, 6e-6, 1e-6), scale=(1e-6, 1e-6, 0), size=3)) for _ in range(N)]
# Setting first muon to full lifetime for debugging
# particles[0].lifetime = Muon.TIME_SCALE[-1]

dipole_data = func.load_run(run_name, files=["dipoles"])
dipole_array = dipole_data["dipoles"]["dipoles"]

fields = np.zeros((N, 3), dtype=float)
for i, p in enumerate(particles):
    for d in dipole_array:
        p.feel_dipole(d)
    fields[i] = p.field

magnitudes = np.array([func.get_mag(f) for f in fields])
field_dict = {"total": magnitudes, "x": fields[:, 0], "y": fields[:, 1], "z": fields[:, 2]}

# Get each muons polarisation
relaxations = np.array([p.full_relaxation(p.field, life_limit=False) for i, p in enumerate(particles)])

# Normalise sum
overall = np.nansum(relaxations, axis=0) / N

popt, pcov = curve_fit(static_GKT, Muon.TIME_SCALE, overall, p0=1e-4)

# Setup subplots
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
ax1 = plt.subplot2grid((2, 3), (0, 2))
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (1, 1))
ax4 = plt.subplot2grid((2, 3), (1, 2))

field_axes = (ax1, ax2, ax3, ax4)
# Plot individual lines if N is small
if len(relaxations) < 100:
    for i in range(N):
        ax0.plot(Muon.TIME_SCALE, relaxations[i], alpha=0.5, lw=0.5)

# Plot overall relaxation
ax0.plot(Muon.TIME_SCALE, overall, lw=2, c="k", alpha=0.7, label="Model")
ax0.plot(Muon.TIME_SCALE, static_GKT(Muon.TIME_SCALE, *popt), c="r", label="Curve fit")

ax0.legend(loc="upper right")
ax0.set_xlim(0, Muon.TIME_SCALE[-1])
ax0.grid()
ax0.set_title("Relaxation function from dipole grid")
ax0.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))

ax1.set_title("Magnitudes of overall field")

for sub_ax, field in zip(field_axes, field_dict.keys()):
    sub_ax.hist(field_dict[field], bins=100)
    sub_ax.set_title(f"Magnitudes of {field}")
    sub_ax.set_xlabel("Field strength (T)")
    sub_ax.set_ylabel("Frequency")
    sub_ax.grid()
    sub_ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
# Add legend
plt.tight_layout(pad=1)

# print(f"Actual width: {random_width}")
print(f"Calculated width: {popt} +- {pcov}")

plt.show()

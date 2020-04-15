# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import modules.functions as func
from modules.muon import Muon
from modules.dipole import Dipole
from modules.model_equations import static_GKT
from modules.multi_process import MP_fields
import time


############################################
def calc_single_particle(input):
    particle, dipole_array = input
    for d in dipole_array:
        particle.feel_dipole(d)
    return particle


def calc_fields(*args):
    particle_chunk, dipoles = args
    chunk_fields = []
    for i, p in enumerate(particle_chunk):
        for d in dipoles:
            p.feel_dipole(d)
        chunk_fields.append(p.field)
        if i % int((len(particle_chunk) / 10)) == 0:
            print(f"{i}/{len(particle_chunk)} done")
    return chunk_fields


np.random.seed(3)

N = 500
run_name = "50x50_R_0"
calculate = True
locations = np.random.normal(loc=(15e-6, 15e-6, 15e-6), scale=(3e-6, 3e-6, 0), size=(N, 3))
particles = np.array([Muon(location=locations[i]) for i in range(N)])

dipole_data = func.load_run(run_name, files=["dipoles"])
dipole_array = dipole_data["dipoles"]["dipoles"]

fields = np.zeros((N, 3), dtype=float)

fig, ax = plt.subplots()
fig, ax =

# Only perform calculation when necessary
if calculate:
    print("Starting multiprocessing...")
    start = time.time()
    MP_fields(run_name, particles, dipole_array)
    end = time.time()
    print(f"Time taken: {end - start}")

    print("Starting single core...")
    start1 = time.time()
    # calc_fields(particles, dipole_array)
    end1 = time.time()
    print(f"Time taken: {end1 - start1}")

#
# fields = func.load_run(run_name, files=["muon_fields"])
# fields = np.array(fields["muon_fields"]["muon_fields"])
# print(np.shape(fields))
#
# magnitudes = np.array([func.get_mag(f) for f in fields])
# field_dict = {"total": magnitudes, "x": fields[:, 0], "y": fields[:, 1], "z": fields[:, 2]}
#
# # Get each muons polarisation
# relaxations = np.array([p.full_relaxation(fields[i], life_limit=False) for i, p in enumerate(particles)])
#
# # Normalise sum
# overall = np.nansum(relaxations, axis=0) / N
#
# popt, pcov = curve_fit(static_GKT, Muon.TIME_SCALE, overall, p0=1e-4)
#
# # Setup subplots
# ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
# ax1 = plt.subplot2grid((2, 3), (0, 2))
# ax2 = plt.subplot2grid((2, 3), (1, 0))
# ax3 = plt.subplot2grid((2, 3), (1, 1))
# ax4 = plt.subplot2grid((2, 3), (1, 2))
#
# field_axes = (ax1, ax2, ax3, ax4)
# # Plot individual lines if N is small
# if len(relaxations) < 100:
#     for i in range(N):
#         ax0.plot(Muon.TIME_SCALE, relaxations[i], alpha=0.5, lw=0.5)
#
# # Plot overall relaxation
# ax0.plot(Muon.TIME_SCALE, overall, lw=2, c="k", alpha=0.7, label="Model")
# ax0.plot(Muon.TIME_SCALE, static_GKT(Muon.TIME_SCALE, *popt), c="r", label="Curve fit")
#
# ax0.legend(loc="upper right")
# ax0.set_xlim(0, Muon.TIME_SCALE[-1])
# ax0.grid()
# ax0.set_title("Relaxation function from dipole grid")
# ax0.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))
#
# ax1.set_title("Magnitudes of overall field")
#
# for sub_ax, field in zip(field_axes, field_dict.keys()):
#     sub_ax.hist(field_dict[field], bins=100)
#     sub_ax.set_title(f"Magnitudes of {field}")
#     sub_ax.set_xlabel("Field strength (T)")
#     sub_ax.set_ylabel("Frequency")
#     sub_ax.grid()
#     sub_ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
# # Add legend
# plt.tight_layout(pad=1)
#
# # print(f"Actual width: {random_width}")
# print(f"Calculated width: {popt} +- {pcov}")
#
# plt.show()

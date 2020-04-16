import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


import modules.functions as func
from modules.muon import Muon
from modules.grid import Grid
from modules.ensemble import Ensemble
from modules.multi_process import MP_fields
from modules.model_equations import static_GKT


#########################################
LOAD_OBJECTS = False
CALCULATE = False
RUN_NAME = "15X15_R_1"

if LOAD_OBJECTS: # Load from file
    island_grid = Grid(run_name=RUN_NAME, load_only=True)
    particles = Ensemble(run_name=island_grid.run_name, load_only=True)

else: # Calculate and save grid and ensemble
    NUM_MUONS = 20000
    island_grid = Grid()
    RUN_NAME = island_grid.run_name
    particles = Ensemble(N=NUM_MUONS, run_name=RUN_NAME)

if CALCULATE:
    print("Starting multiprocessing...")
    start = time.time()
    MP_fields(RUN_NAME, particles.muons, island_grid.islands)
    end = time.time()
    print(f"Time taken: {end - start}")

fields, field_dict = func.load_fields(RUN_NAME)
# Get each muons polarisation
particles.set_relaxations(fields)
# Normalise sum
overall = np.nansum(particles.relaxations, axis=0) / particles.N

# CURVE FIT
popt, pcov = curve_fit(static_GKT, Muon.TIME_SCALE, overall, p0=1e-4)


# Setup subplots
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
ax1 = plt.subplot2grid((2, 3), (0, 2))
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (1, 1))
ax4 = plt.subplot2grid((2, 3), (1, 2))

field_axes = (ax1, ax2, ax3, ax4)
# Plot individual lines if N is small
if len(particles.relaxations) < 100:
    for i in range(N):
        ax0.plot(Muon.TIME_SCALE, particles.relaxations[i], alpha=0.5, lw=0.5)

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
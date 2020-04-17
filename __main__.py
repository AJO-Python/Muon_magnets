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

RUN_NAME = "15X15_R_4"

if LOAD_OBJECTS:  # Load from file
    island_grid = Grid(run_name=RUN_NAME, load_only=True)
    particles = Ensemble(run_name=island_grid.run_name, load_only=True)

else:  # Calculate and save grid and ensemble
    NUM_MUONS = 20000
    SPREAD_VALUES = {"x_width": 10e-6, "y_width": 10e-6, "z_width": 10e-6,
                     "x_mean": 0, "y_mean": 0, "z_mean": 100e-6}
    island_grid = Grid()
    RUN_NAME = island_grid.run_name
    particles = Ensemble(N=NUM_MUONS, spread_values=SPREAD_VALUES, run_name=RUN_NAME)
    particles.set_generic("spin_dir", [1, 0, 0])
    particles.calculate_fields(island_grid)
    particles.load_fields()


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
ax0.set_xlim(0, 20e-6)
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
plt.savefig(f"data/{RUN_NAME}/Relax_fields.png")
# print(f"Actual width: {random_width}")
print(f"Calculated width: {popt} +- {pcov}")

fig = plt.figure()
fig.suptitle(RUN_NAME)

angle_ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
grid_ax = plt.subplot2grid((2, 2), (0, 0))
XZ_dist_ax = plt.subplot2grid((2, 2), (0, 1))

angle_ax.hist(island_grid.angles, bins=36)
angle_ax.set_xlabel("Rotation of island (degrees)")
angle_ax.set_ylabel("Frequency")
angle_ax.set_title("Angle distribution of Dipoles")

XZ_dist_ax.scatter(particles.loc[:, 0::3], particles.loc[:, 2::3], s=1, alpha=0.5)
XZ_dist_ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
XZ_dist_ax.set_aspect("equal")
XZ_dist_ax.set_xlim(-30e-6, 30e-6)
XZ_dist_ax.set_ylim(370e-6, 430e-6)
XZ_dist_ax.set_xlabel("X locations")
XZ_dist_ax.set_ylabel("Z locations")
XZ_dist_ax.set_title("Z and X distributions of muons")

grid_ax.set_aspect("equal")
island_grid.set_generic("line_width", 1)
fig, grid_ax = island_grid.show_on_plot(fig, grid_ax)
fig, grid_ax = particles.show_on_plot(fig, grid_ax, thin=3)
grid_ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))

fig.tight_layout(pad=0.1)
fig.savefig(f"data/{RUN_NAME}/visualise.png")

plt.show()

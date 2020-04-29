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

RUN_NAME = "Z_slice_10um_10XYspread_U"

if LOAD_OBJECTS:  # Load from file
    island_grid = Grid(run_name=RUN_NAME, load_only=True)
    particles = Ensemble(run_name=island_grid.run_name, load_only=True)
    particles.load_fields()

else:  # Calculate and save grid and ensemble
    NUM_MUONS = 20000
    SPREAD_VALUES = {"x_width": 10e-6, "y_width": 10e-6, "z_width": 0,
                     "x_mean": 0, "y_mean": 0, "z_mean": 10e-6}
    # run_name = "Z_slice_10um_10XYspread_U"
    island_grid = Grid()
    RUN_NAME = island_grid.run_name
    particles = Ensemble(N=NUM_MUONS, loc_spread_values=SPREAD_VALUES, run_name=RUN_NAME)
    particles.calculate_fields(island_grid)
    # Reload ensemble after calculating fields
    # Ensures that fields and locations stay in the
    # same order after multiprocessing
    particles = Ensemble(run_name=RUN_NAME, load_only=True)

# Get each muons polarisation
particles.set_relaxations()

particles.plot_relax_fields(save=True)
fig = plt.figure(figsize=func.set_fig_size(subplots=(3, 2)))
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

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

RUN_NAME = "20X20_R_2"

if LOAD_OBJECTS:  # Load from file
    island_grid = Grid(run_name=RUN_NAME, load_only=True)
    particles = Ensemble(run_name=island_grid.run_name, load_only=True)

else:  # Calculate and save grid and ensemble
    NUM_MUONS = 20000
    SPREAD_VALUES = {"x_width": 10e-6, "y_width": 10e-6, "z_width": 10e-6,
                     "x_mean": 0, "y_mean": 0, "z_mean": 100e-6}
    island_grid = Grid()
    RUN_NAME = island_grid.run_name
    particles = Ensemble(N=NUM_MUONS, run_name=RUN_NAME)
    # particles.set_generic("spin_dir", [1, 0, 0])
    particles.calculate_fields(island_grid)
    particles.load_fields()
    particles.save_ensemble()

# Get each muons polarisation
particles.set_relaxations()

# Plot graphs
particles.plot_relax_fields(save=True)
particles.plot_distribution(grid=island_grid, save=True)

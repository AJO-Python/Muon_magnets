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

RUN_NAME = "random_fields"
NUM_MUONS = 20_000
SPREAD_VALUES = {"x_width": 10e-6, "y_width": 10e-6, "z_width": 10e-6,
                 "x_mean": 0, "y_mean": 0, "z_mean": 0}
particles = Ensemble(N=NUM_MUONS, spread_values=SPREAD_VALUES, run_name=RUN_NAME)

fields = np.random.normal(0, 1e-3, size=(NUM_MUONS, 3))

magnitudes = np.array([func.get_mag(f) for f in fields])
field_dict = {"total": magnitudes, "x": fields[:, 0], "y": fields[:, 1], "z": fields[:, 2]}

particles.set_generic("fields", fields)

particles.set_relaxations(fields)

func.plot_relaxations(particles, RUN_NAME, "ZF_relax", field_dict)

particles.add_field(np.array([1e-3, 0, 0]))
fieldsX = particles.fields
field_dictX = particles.field_dict
func.plot_relaxations(particles, RUN_NAME, "add_X_relax", field_dictX)

plt.show()

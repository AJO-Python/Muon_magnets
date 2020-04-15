import numpy as np
import matplotlib.pyplot as plt

import modules.functions as func
from modules.grid import Grid
from modules.ensemble import Ensemble
from modules.

#########################################
print("Loading config file 'dipole_array_config.txt'...")
run_name = make_dipole_grid()
print(f"Dipoles stored in: data/{run_name}")

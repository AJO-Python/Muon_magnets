import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import modules.functions as func
import modules.grid as grid

#########################################
print("Loading config file 'dipole_array_config.txt'...")
run_name = make_dipole_grid()
print(f"Dipoles stored in: data/{run_name}")

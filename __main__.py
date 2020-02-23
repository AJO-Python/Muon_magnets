import numpy as np
import matplotlib.pyplot as plt

import Modules.functions as func
import Modules.muon as mu
import Modules.dipole as dip
import Modules.grid as grid

Nx, Ny = 10, 10
side_length = 2e-3
x_array = np.linspace(-side_length, side_length, Nx)
y_array = np.linspace(-side_length, side_length, Ny)

dipoles_aligned = dip.create_dipole_grid(x_array, y_array,
                                         dipole_spacing=5,
                                         random_angle=False,
                                         strength=1e-2,
                                         buffer=1)
print("Test")
print(dipoles_aligned)
# field = set_field_values(x_array, y_array, dipoles_aligned, resolution=2)
# field = [dipoles_aligned.get_point(*coord) for coord in dipoles_aligned.all_coords()]
# print(field.get_axis_values("all"))

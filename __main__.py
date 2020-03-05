import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import modules.functions as func
import modules.grid as grid


def create_field_grid(runname,
                      x_points=20, y_points=20,
                      field_spacing=1e-6, dipole_spacing=3e-6,
                      rand=False):
    dipole_grid = grid.Grid(x_points, y_points)
    # Find limits of dipoles so we know where to plot the field lines
    dipole_grid.fill_with_dipoles(spacing=dipole_spacing, angle=0,
                                  random_angle=rand)
    x_max, y_max = dipole_grid.real_size
    real_locs_x = np.arange(0, x_max, field_spacing)
    real_locs_y = np.arange(0, y_max, field_spacing)
    locations = np.array([real_locs_x, real_locs_y])

    # Get field values over field array
    Ex, Ey = dipole_grid.fill_field_values(real_locs_x, real_locs_y)

    # Save field values
    func.save_array(f"E_uniform_{runname}", Ex=Ex, Ey=Ey)
    # Save locations
    func.save_array(f"RL_uniform_{runname}", loc=locations)
    # Save grid object
    func.save_object(f"DG_uniform_{runname}", dipole_grid)


#%%
#############################################################
# RUN PARAMETERS
x_points = 3
y_points = 1
field_sample_spacing = 2e-7
dipole_spacing = 3e-6
random = False
runname = "5x2_1e7_U"

create_field_grid(runname, x_points=x_points, y_points=y_points,
                  field_spacing=field_sample_spacing,
                  dipole_spacing=dipole_spacing,
                  rand=random)
#%%
#############################################################

E_field = np.load(f"./data/E_uniform_{runname}.npz")
print(E_field.files)
print("----")
locations = np.load(f"./data/RL_uniform_{runname}.npz")
print(locations.files)
#%%
dipole_grid = func.load_object(f"DG_uniform_{runname}")
x_max, y_max = dipole_grid.real_size
x_max, y_max = max(real_locs_x), max(real_locs_y)

#%%
#############################################################
# PLOTTING
#############################################################

fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the streamlines with an appropriate colormap and arrow style
color = 2 * np.log(np.hypot(Ex, Ey))
ax.streamplot(real_locs_x, real_locs_y, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=5, arrowstyle='->', arrowsize=1.5)

ells = [Ellipse(xy=d.location,
                width=700e-9, height=1.6e-6,
                angle=d.orientation_d + 90)
        for d in dipole_grid.all_values()]

for ellipse in ells:
    ax.add_artist(ellipse)
# for d in dipole_grid.all_values():
#    ax.add_artist(Circle(d.location, 0.0000001, color=charge_colors[1 > 0]))
ax.set_xlabel('$x (m)$')
ax.set_ylabel('$y (m)$')
ax.set_xlim(0 - (x_max / 20), x_max + (x_max / 20))
ax.set_ylim(0 - (x_max / 20), y_max + (y_max / 20))
ax.set_title(r"$\bar{B}$ lines from an array of nano-islands")
# ax.set_xlim(-1e-6, 10e-6)
# ax.set_ylim(-1e-6, 10e-6)
ax.set_aspect('equal')
plt.savefig(f"images/dipoles/real_grid_{runname}.png", bbox_inches="tight")
# plt.colorbar()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import Modules.functions as func
import Modules.muon as mu
import Modules.dipole as dip
import Modules.grid as grid
import model_equations


def create_field_grid(x_points=20, y_points=20,
                      spacing=1e-6):
    dipole_grid = grid.Grid(x_points, y_points)
    # Find limits of dipoles so we know where to plot the field lines
    dipole_grid.fill_with_dipoles(spacing=3e-6, angle=0,
                                  random_angle=False,
                                  full=True)
    x_max, y_max = dipole_grid.real_size
    real_locs_x = np.arange(0, x_max, spacing)
    real_locs_y = np.arange(0, y_max, spacing)

    # Get field values over field array
    Ex, Ey = dipole_grid.fill_field_values(real_locs_x, real_locs_y)

    # Save field values
    func.save_array("Ex_uniform_1", Ex)
    func.save_array("Ey_uniform_1", Ey)
    # Save locations
    func.save_array("RL_uniform_1", [real_locs_x, real_locs_y])
    # Save grid object
    func.save_object("DG_uniform_1", dipole_grid)


#############################################################
x_points = 10
y_points = 10
field_sample_spacing = 0.25e-6

# create_field_grid(x_points=x_points, y_points=y_points,
#                  spacing=field_sample_spacing)

Ex = np.loadtxt("./Results/Ex_uniform_1.txt")
Ey = np.loadtxt("./Results/Ey_uniform_1.txt")
[real_locs_x, real_locs_y] = np.loadtxt("./Results/RL_uniform_1.txt")
dipole_grid = func.load_object("DG_uniform_1")
x_max, y_max = dipole_grid.real_size
x_max, y_max = max(real_locs_x), max(real_locs_y)

# print(f"real size: {dipole_grid.real_size}")
# print(f"size: {dipole_grid.size}")
# print(f"x, y max: {x_max}, {y_max}")
# print(f"Num. of dipoles: {dip.Dipole.count}")
# print("Location of dipoles:")
# for d in dipole_grid.all_values():
#     print(d.location)


fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the streamlines with an appropriate colormap and arrow style
color = 2 * np.log(np.hypot(Ex, Ey))
ax.streamplot(real_locs_x, real_locs_y, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

# Add filled circles for the charges themselves
charge_colors = {True: '#aa0000', False: '#0000aa'}

ells = [Ellipse(xy=d.location,
                width=700e-9, height=1.6e-6,
                angle=d.orientation_d + 90)
        for d in dipole_grid.all_values()]

for ellipse in ells:
    ax.add_artist(ellipse)
# for d in dipole_grid.all_values():
#    ax.add_artist(Circle(d.location, 0.0000001, color=charge_colors[1 > 0]))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(0 - (x_max / 20), x_max + (x_max / 20))
ax.set_ylim(0 - (x_max / 20), y_max + (y_max / 20))
# ax.set_xlim(-1e-6, 10e-6)
# ax.set_ylim(-1e-6, 10e-6)
ax.set_aspect('equal')
plt.savefig("Images/Dipoles/real_grid_v1.png", bbox_inches="tight")
plt.show()

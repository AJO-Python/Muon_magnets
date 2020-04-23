# -*-coding: UTF-8-*-
import numpy as np
import matplotlib.pyplot as plt

from modules.grid import Grid
from modules.island import Island


def setup_field(width, height, nx, ny):
    """
    :param tuple max_loc: (X, Y) of furthest dipole
    :param float edge_buffer: Distance around edge of grid to calculate field for
    :param int nx: Number of field x points
    :param int ny: Number of field y points
    :rtype: dict
    :return: Dictionary containing coordinates for a field over the dipole array
    """
    edge_buffer_x = width / 10
    edge_buffer_y = height / 10
    # Determine region to calculate field lines/plot over
    field_region = [[(-width / 2) - edge_buffer_x, (width / 2) + edge_buffer_x],
                    [(-height / 2) - edge_buffer_y, (height / 2) + edge_buffer_y]]
    field_locations = {"x_vals": np.linspace(*field_region[0], nx),
                       "y_vals": np.linspace(*field_region[1], ny)}
    return field_locations


island_1 = Island(orientation=0, strength=8e-8,
                  location=[0, 0, 0], size=[1.6e-6, 700e-9])
island_2 = Island(orientation=0, strength=8e-8,
                  location=[4e-6, 0, 0], size=[1.6e-6, 700e-9])
islands = [island_1, island_2]

field_locations = setup_field(30e-6, 30e-6, 100, 100)

x_locs = field_locations["x_vals"]
y_locs = field_locations["y_vals"]

X, Y = np.meshgrid(x_locs, y_locs)

# Ex = np.zeros(np.shape(X))
# Ey = np.zeros_like(Ex)
# Ez = np.zeros_like(Ex)
# for j in range(len(X)):
#     for i in range(len(Y)):
#         for isle in islands:
#             ex, ey, ez = isle.get_mag_field([X[0, j], Y[i, 0], 10e-6])
#             Ex[i][j] += np.add(Ex[i][j], ex)
#             Ey[i][j] += np.add(Ey[i][j], ey)
#             Ex[i][j] += np.add(Ez[i][j], ez)


for i, x in enumerate(x_locs):
    for j, y in enumerate(y_locs):
        for isle in islands:
            ex, ey, _ = isle.get_mag_field([x, y, 10e-6])
            Ex[j][i] += ex
            Ey[j][i] += ey

color = 2 * np.log(np.hypot(Ex, Ey))

fig, ax = plt.subplots()
ax.streamplot(x_locs, y_locs, Ex, Ey,
              color=color,
              linewidth=1,
              cmap=plt.cm.inferno,
              density=2,
              arrowstyle='->',
              arrowsize=1.5)
ax.set_aspect("equal")
for isle in islands:
    ax.add_artist(isle.get_outline())
    ax.add_artist(isle.get_moment_arrow())

ax.set_xlim(min(x_locs), max(x_locs))
ax.set_ylim(min(y_locs), max(y_locs))
ax.ticklabel_format(axis="both", style="sci", scilimits=(-6, -6))
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:26:49 2020

@author: joshowen121
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from modules.dipole import Dipole
import modules.functions as func


def get_mag_field(start, moment, x, y):
    mag_perm = 10e-7  # Cancel constant terms to get mag_perm as only constant
    relative_loc = np.array([x-start[0], y-start[1]])
    magnitude = np.array([func.get_mag(loc) for loc in relative_loc])
    moments = np.zeros_like(magnitude, dtype=np.ndarray)
    moments.fill(moment)
    #dots = np.dot(moments, relative_loc)
    dots = np.zeros_like(x)
    for i, (xs, ys) in enumerate(zip(*relative_loc)):
        for j, (rx, ry) in enumerate(zip(xs, ys)):
            dots[i][j] = rx*moment[0] + ry*moment[1]
    return mag_perm * (
               (3 * relative_loc * (dots)
                / (magnitude ** 5))
                - (moments / (magnitude ** 3))
             )

def fill_field_values(dipoles, x_locs, y_locs):
    x_len = len(x_locs)
    y_len = len(y_locs)
    Ex = np.zeros([x_len, y_len])
    Ey = np.zeros([x_len, y_len])
    for i, x in enumerate(x_locs):
        print(f"Calculating row... {i}/{x_len - 1}")
        for j, y in enumerate(y_locs):
            for dipole in dipoles:
                ex, ey = dipole.get_mag_field([x, y])
                Ex[j][i] += ex
                Ey[j][i] += ey
    return Ex, Ey


width = 2
height = 2
spacing = 3e-6

def main():
    """

    """
    width, height, spacing, orientation = np.loadtxt("dipole_array_config.txt")

    # Create coordinates for grid
    coords = [(x, y) for x in range(width) for y in range(height)]
    dipole_count = len(coords)

    # Create array of dipoles
    dipole_array = np.empty(dipole_count, dtype=object)
    angles = np.random.uniform(0, 360, dipole_count)
    for i, coord in enumerate(coords):
        dipole_array[i] = Dipole(orientation=angles[i],
                                coord=coord,
                                location=(coord[0]*spacing, coord[1]*spacing))

    # Get size of grid using further dipole
    x_max, y_max = dipole_array[-1].location

    # Determine region to calculate field lines/plot over
    field_region = [[0-spacing, x_max+spacing], [0-spacing, y_max+spacing]]
    nx, ny = 50, 50

    field_region = [[0-spacing, x_max+spacing], [0-spacing, y_max+spacing]]
    x_vals = np.linspace(*field_region[0], nx)
    y_vals = np.linspace(*field_region[1], ny)

    Ex, Ey = fill_field_values(dipole_array, x_vals, y_vals)

    func.save_array("test", ex=Ex, ey=Ey)


# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111)

# Set colourscale and plot streamplot
color = 2 * np.log(np.hypot(Ex, Ey))
ax.streamplot(x_vals, y_vals, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

# Add ellipses to mark dipoles
ells = [Ellipse(xy=dipole.location,
                width=700e-9, height=1.6e-6,
                angle=dipole.orientation_d + 90) for dipole in dipole_array]
for ellipse in ells:
    ax.add_artist(ellipse)

# Set graph parameters
ax.set_xlabel('$x (m)$')
ax.set_ylabel('$y (m)$')
ax.set_xlim(field_region[0])
ax.set_ylim(field_region[1])
ax.set_title("Test for grid")
ax.set_aspect('equal')
plt.grid()
#plt.savefig(f"images/dipoles/real_grid_{runname}.png", bbox_inches="tight")
plt.show()

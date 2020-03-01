#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:54:43 2020

@author: joshowen121
"""
import modules.functions as func
from modules.dipole import Dipole

def main():
    """

    """
    data = func.load_config("dipole_array_config")
    spacing = data["spacing"]

    # Create coordinates for grid
    coords = [(x, y) for x in range(int(data["width"])) for y in range(int(data["height"]))]
    dipole_count = len(coords)

    # Create array of dipoles
    dipole_array = np.empty(dipole_count, dtype=object)

    # Set dipole angles
    if data["random_orientation"]:
        angles = np.random.uniform(0, 360, dipole_count)
    else:
        angles = np.full_like(
                    dipole_array, fill_value=data["angle"], dtype=float)

    # Fill array with dipoles with $spacing between each one
    for i, coord in enumerate(coords):
        dipole_array[i] = Dipole(orientation=angles[i],
                                coord=coord,
                                location=(coord[0]*spacing, coord[1]*spacing))

    # Get maximum location of grid using furthest dipole
    x_max, y_max = dipole_array[-1].location

    # Determine region to calculate field lines/plot over
    field_region = [[0-spacing, x_max+spacing], [0-spacing, y_max+spacing]]
    nx, ny = 50, 50

    field_region = [[0-spacing, x_max+spacing], [0-spacing, y_max+spacing]]
    x_vals = np.linspace(*field_region[0], nx)
    y_vals = np.linspace(*field_region[1], ny)

    Ex, Ey = fill_field_values(dipole_array, x_vals, y_vals)

    func.save_array("test", ex=Ex, ey=Ey)
    func.save_array("test_vals", xval=x_vals, yval=y_vals)


main()


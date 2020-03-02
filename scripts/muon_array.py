#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:26:49 2020

@author: joshowen121
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from modules.dipole import Dipole
import modules.functions as func


def make_grid_and_field():
    """
    Creates and saves an array of dipoles and corresponding fields/locations
    Loads from "dipole_array_config.txt"
    Saves to "Muon_magnets/data/$run_name/..."

    :rtype: str
    :return: run_name for use with "load_run()"
    """
    # Load configuration for run
    data = func.load_config("dipole_array_config")
    width = int(data["width"])
    height = int(data["height"])
    r_angle = 'R' if data['random_orientation'] else 'U'

    # Set $run_name and create unique run directory
    run_name = check_run_name(f"{width}x{height}_{r_angle}_0")

    # Create coordinates for grid
    coords = [(x, y) for x in range(width) for y in range(height)]
    dipole_count = len(coords)

    # Create array of dipoles (one for each coordinate)
    dipole_array = np.empty(dipole_count, dtype=object)

    # Set dipole angles
    if data["random_orientation"]:
        angles = np.random.uniform(0, 360, dipole_count)
    else:
        angles = np.full_like(
            dipole_array, fill_value=data["angle"], dtype=float)

    # Fill array with dipoles with $spacing between each point
    spacing = data["spacing"]
    for i, coord in enumerate(coords):
        dipole_array[i] = Dipole(orientation=angles[i],
                                 coord=coord,
                                 location=(coord[0] * spacing, coord[1] * spacing))

    # Setup field grid and calculate values
    nx, ny = data["nx"], data["ny"]
    field_locations = setup_field(max_loc=dipole_array[-1].location,
                                  edge_buffer=spacing,
                                  nx=nx, ny=ny)
    field_values = fill_field_values_2d(dipole_array, **field_locations)

    # Save data
    func.save_array(run_name, "dipoles", dipoles=dipole_array)
    func.save_array(run_name, "fields", **field_values)
    func.save_array(run_name, "locations", **field_locations)

    return run_name


def check_run_name(run_name):
    """
    :param str run_name: Proposed name of run
    :rtype: str
    :return: Make unique directory and return name
    """
    made_dir = False
    dir_count = 0
    while not made_dir:
        try:
            os.makedirs(f"../data/{run_name}")
            made_dir = True
        except OSError:
            dir_count += 1
            run_name = run_name[:-1] + str(dir_count)
    return run_name


def setup_field(max_loc, edge_buffer, nx, ny):
    """
    :param tuple max_loc: (X, Y) of furthest dipole
    :param float edge_buffer: Distance around edge of grid to calculate field for
    :param int nx: Number of field x points
    :param int ny: Number of field y points
    :rtype: dict
    :return: Dictionary containing coordinates for a field over the dipole array
    """
    x_max, y_max = max_loc
    # Determine region to calculate field lines/plot over
    field_region = [[0 - edge_buffer, x_max + edge_buffer], [0 - edge_buffer, y_max + edge_buffer]]
    field_locations = {"x_vals": np.linspace(*field_region[0], nx),
                       "y_vals": np.linspace(*field_region[1], ny)}
    return field_locations


def fill_field_values_2d(dipoles, x_vals=[], y_vals=[]):
    """
    :param array dipoles: array of dipoles
    :param array x_locs: x locations to calculate field at
    :param array y_locs: y locations to calculate field at
    :rtype: dict
    :return: X and Y field values for a meshgrid of x_locs and y_locs
    """
    x_len = len(x_vals)
    y_len = len(y_vals)
    Ex = np.zeros([x_len, y_len])
    Ey = np.zeros([x_len, y_len])
    for i, x in enumerate(x_vals):
        print(f"Calculating row... {i}/{x_len - 1}")
        for j, y in enumerate(y_vals):
            for dipole in dipoles:
                if dipole.location == (x, y):
                    continue
                ex, ey = dipole.get_mag_field([x, y])
                Ex[j][i] += ex
                Ey[j][i] += ey
    return {"Ex": Ex, "Ey": Ey}


def load_run(run_name):
    """
    :param str run_name: Folder run is saved to
    :rtype: Dict
    :return: Three dictionaries with dipole, field, and location data
    """
    dipole_data = np.load(f"../data/{run_name}/dipoles.npz", allow_pickle=True)
    print(f"Loaded dipoles")

    field_data = np.load(f"../data/{run_name}/fields.npz")
    print(f"Loaded fields")

    loc_data = np.load(f"../data/{run_name}/locations.npz")
    print(f"Loaded from locations")

    return dipole_data, field_data, loc_data

if __name__ == "__main__":

    run_name = make_grid_and_field()

    dipole_data, field_data, loc_data = load_run(run_name)
    dipole_array = dipole_data["dipoles"]

    plot_density = 1

    #PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Set colourscale and plot streamplot
    color = 2 * np.log(np.hypot(field_data["Ex"], field_data["Ey"]))

    ax.streamplot(loc_data["x_vals"],
                  loc_data["y_vals"],
                  field_data["Ex"],
                  field_data["Ey"],
                  color=color,
                  linewidth=1,
                  cmap=plt.cm.inferno,
                  density=plot_density,
                  arrowstyle='->',
                  arrowsize=1.5)

    # Add ellipses to mark dipoles
    ells = [Ellipse(xy=dipole.location,
                    width=700e-9, height=1.6e-6,
                    angle=dipole.orientation_d + 90) for dipole in dipole_array]
    for ellipse in ells:
        ax.add_artist(ellipse)

    # Set graph parameters
    ax.set_xlabel('$x (m)$')
    ax.set_ylabel('$y (m)$')
    ax.set_xlim(min(loc_data["x_vals"]), max(loc_data["x_vals"]))
    ax.set_ylim(min(loc_data["y_vals"]), max(loc_data["y_vals"]))
    ax.set_title(f"{run_name} grid")
    ax.set_aspect('equal')
    plt.grid()
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-6, -6))
    plt.savefig(f"../images/dipoles/{run_name}.png", bbox_inches="tight")
    plt.show()

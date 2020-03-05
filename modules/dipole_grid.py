#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from modules.dipole import Dipole
import modules.functions as func


def make_dipole_grid():
    """
    Creates and saves an array of dipoles on a grid
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
    run_name = check_run_name(f"{width}x{height}_{r_angle}")

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
                                 location=(coord[0] * spacing, coord[1] * spacing),
                                 strength=data["strength"])

    func.save_array(run_name, "dipoles", dipoles=dipole_array)
    return run_name


def check_run_name(run_name):
    """
    :param str run_name: Proposed name of run
    :rtype: str
    :return: Make unique directory and return name
    """
    made_dir = False
    dir_count = 0
    run_name = run_name + "_" + str(dir_count)
    while not made_dir:
        try:
            os.makedirs(f"../data/{run_name}")
            made_dir = True
        except OSError:
            dir_count += 1
            if dir_count <= 10:
                run_name = run_name[:-1] + str(dir_count)
            else:
                run_name = run_name[:-2] + str(dir_count)
    return run_name


if __name__ == "__main__":
    print("Loading config file 'dipole_array_config.txt'...")
    run_name = make_dipole_grid()
    print(f"Dipoles stored in: ../data/{run_name}")
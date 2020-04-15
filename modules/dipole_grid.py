# -*- coding: utf-8 -*-
import os
import numpy as np

from modules.dipole import Dipole
from modules.island import Island
import modules.functions as func


def make_dipole_grid(config_file="", **kwargs):
    """
    Creates and saves an array of dipoles on a grid
    Loads from "dipole_array_config.txt"
    Saves to "Muon_magnets/data/$run_name/..."

    :rtype: str
    :return: run_name for use with "load_run()"
    """
    # Load configuration for run
    if not config_file:
        # print("Using default value")
        config_file = "dipole_array_config"
    data = func.load_config(config_file)
    width = int(data["width"])
    height = int(data["height"])
    r_angle = 'R' if data['random_orientation'] else 'U'

    # Set $run_name and create unique run directory
    run_name = check_run_name(f"{width}x{height}_{r_angle}")

    # Create coordinates for grid
    coords = [(x, y, 0) for x in range(width) for y in range(height)]
    dipole_count = len(coords)

    # Create array of dipoles (one for each coordinate)
    dipole_array = np.empty(dipole_count, dtype=object)
    angles = set_angles(dipole_array, data, data["random_orientation"])
    # Fill array with dipoles with $spacing between each point
    spacing = data["spacing"]
    for i, coord in enumerate(coords):
        dipole_array[i] = Island(orientation=angles[i],
                                 coord=coord,
                                 location=(coord[0] * spacing, coord[1] * spacing, 0),
                                 strength=data["strength"])

    # Can return the array for testing the function
    for key, value in kwargs.items():
        if key == "testing" and value == True:
            print("In testing mode")
            return dipole_array

    # If not testing save array to file
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
            os.makedirs(f"data/{run_name}")
            made_dir = True
        except OSError:
            dir_count += 1
            if dir_count <= 10:
                run_name = run_name[:-1] + str(dir_count)
            else:
                run_name = run_name[:-2] + str(dir_count)
    return run_name


def set_angles(dipoles, data, random):
    """
    Create array of angles to assign to dipoles
    :param dipoles: array of dipoles
    :param random:
    :return:
    """
    # Set dipole angles
    if random:
        angles = np.random.uniform(0, 360, len(dipoles))
    else:
        angles = np.full_like(
            dipoles, fill_value=data["angle"], dtype=float)
    return angles


if __name__ == "__main__":
    print("Loading config file 'dipole_array_config.txt'...")
    run_name = make_dipole_grid()
    print(f"Dipoles stored in: data/{run_name}")

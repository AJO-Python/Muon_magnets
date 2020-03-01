#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:54:43 2020

@author: joshowen121
"""

import numpy as np


def load_config(file_name):
    """
    :param str file_name: Name of file in Muon_magnets/config/{file_name}
    :rtype: dict
    :return: Dictionary of config variables
    """
    # Load file as string
    load_data = np.loadtxt(f"../config/{file_name}.txt",
                      delimiter="\n",
                      unpack=False,
                      dtype=str)
    # Unpack into
    data = {}
    for item in load_data:
        key, value = item.split("=")
        try:
            data[key.strip()] = float(value.strip())
        except ValueError:
            print(f"\"{item}\" is not float. Trying bool...")
            data[key.strip()] = bool(value.strip())
    return data

print(load_config("dipole_array_config"))

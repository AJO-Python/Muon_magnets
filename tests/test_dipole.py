# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:16:57 2019

@author: c1672922
"""

import numpy as np
import pytest
from Modules.dipole import Dipole

def test_dipole():
    Dipole.count = 0
    dipole = Dipole(orientation=45, location=[1,1,1], strength=1)
    
    # Check all attributes assigned
    assert hasattr(dipole, "strength")
    assert hasattr(dipole, "location")
    assert hasattr(dipole, "orientation_d")
    assert hasattr(dipole, "orientation_r")
    assert hasattr(dipole, "moment")
    
    assert type(dipole.loc) == np.ndarray
    assert type(dipole.moment) == np.ndarray

    # Check orientation and angle conversion correct
    assert dipole.orientation_d == 45
    assert dipole.orientation_r == np.deg2rad(dipole.orientation_d)
    assert len(dipole.moment) == len(dipole.loc)

    # Check class variables
    assert Dipole.count == 1

def test_mag_field():
    dipole = Dipole(orientation=0, location=[0,0,0], strength=1)
    field = dipole.get_mag_field([2,0,0])
    assert type(field) == np.ndarray
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
import numpy as np
import Modules.functions as func


def test_get_mag():
    # Check basic functionality
    assert func.get_mag(np.array([1, 0, 0])) == 1
    # Check negatives handled
    assert func.get_mag(np.array([-1, 0, 0])) == 1
    # Check 3d space
    assert func.get_mag(np.array([1, 1, 1])) == np.sqrt(3)


def test_get_unit_vector():
    assert np.allclose(func.get_unit_vector(np.array([1, 0, 0])), np.array([1, 0, 0]))     # Basic test
    assert func.get_unit_vector(np.array([10, 0, 0])) == np.array([1, 0, 0])    # Test for higher numbers
    assert func.get_unit_vector(np.array([0, 0, 0])) == np.array([0, 0, 0])     # Test for zeros
    assert func.get_unit_vector(np.array([-1, 0, 0])) == np.array([-1, 0, 0])   # Test negative numbers

def test_get_angle():
    assert func.get_angle(np.array([1, 0, 0]), np.array([0, 1, 0])) == np.pi/2
    assert func.get_angle(np.array([1, 0, 0]), np.array([1, 0, 0])) == 0


def test_detect_asym():
    assert func.detect_asym(1, 0) == 1  # Only forwards
    assert func.detect_asym(0, 1) == -1  # Only backwards
    assert func.detect_asym(1, 1) == 0  # Numerator == 0
    assert func.detect_asym(0, 0) == 0  # Denominator == 0
    assert func.detect_asym(-1, 0) == 1  # One negative
    assert func.detect_asym(-1, -1) == 0  # Both negative


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
import numpy as np
import pytest
import Modules.functions as func



def test_set_mag():
    print("Testing func.set_mag...")
    # Check basic functionality
    assert func.set_mag([1, 0, 0]) == 1
    # Check negatives handled
    assert func.set_mag([-1, 0, 0]) == 1
    # Check 3d space
    assert func.set_mag([1, 1, 1]) == np.sqrt(3)


def test_mag_force():
    print("Testing func.mag_force...")
    q = 1
    v = [1, 0, 0]
    B = [0, 1, 0]

    output_norm = func.mag_force(q, v, B)
    assert list(output_norm) == list([0, 0, 1])

    output_neg = func.mag_force(-q, v, B)
    assert list(output_neg) == list([0, 0, -1])

    v_2 = [2, 0, 0]
    output_double = func.mag_force(q, v_2, B)
    assert list(output_double) == list([0, 0, 2])

"""
def test_larmor_freq():
    print("Testing func.larmor_freq...")
    # Test simple case
    assert func.larmor_freq(1) == Muon.gyro_ratio
    # Test negative handling
    assert func.larmor_freq(-1) == Muon.gyro_ratio
    # Test small numbers
    assert pytest.approx(func.larmor_freq(1e-9)) == 1e-9 * Muon.gyro_ratio
"""

def test_detect_asym():
    print("Testing func.detect_asym...")
    assert func.detect_asym(1, 0) == 1  # Only forwards
    assert func.detect_asym(0, 1) == -1  # Only backwards
    assert func.detect_asym(1, 1) == 0  # Numerator == 0
    assert func.detect_asym(0, 0) == 0  # Denominator == 0
    assert func.detect_asym(-1, 0) == 1  # One negative
    assert func.detect_asym(-1, -1) == 0  # Both negative


def test_decay():
    print("Testing func.decay...")
    pass
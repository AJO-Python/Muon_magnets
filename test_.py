#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
import pytest
import Modules.muon as muon
import Modules.positron as positron
import Modules.functions as func
import numpy as np


def test_get_mag():
    # Check basic functionality
    assert func.get_mag([1, 0, 0]) == 1
    # Check negatives handled
    assert func.get_mag([-1, 0, 0]) == 1
    # Check 3d space
    assert func.get_mag([1, 1, 1]) == np.sqrt(3)


def test_mag_force():
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


def test_larmor_freq():
    # Test simple case
    assert func.larmor_freq(1, 2) == 2
    # Test negative handling
    assert func.larmor_freq(-1, 2) == 2
    # Test small numbers
    assert pytest.approx(func.larmor_freq(1e-9, 0.5e-4)) == 0.5e-13


def test_asym():
    assert func.asym(0, 0) == 0
    assert func.asym(0, 1) == -1
    assert func.asym(1, 1) == 0
    assert func.asym(1, 0) == 1
    assert func.asym(-1, -1) == 0
    assert func.asym(-1, 0) == 1


def test_decay():
    pass

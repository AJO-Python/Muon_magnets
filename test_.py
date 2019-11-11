#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
import pytest
from Modules.muon import Muon
from Modules.positron import Positron
import Modules.functions as func
import numpy as np


def test_get_mag():
    print("Testing func.get_mag...")
    # Check basic functionality
    assert func.get_mag([1, 0, 0]) == 1
    # Check negatives handled
    assert func.get_mag([-1, 0, 0]) == 1
    # Check 3d space
    assert func.get_mag([1, 1, 1]) == np.sqrt(3)


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


def test_larmor_freq():
    print("Testing func.larmor_freq...")
    # Test simple case
    assert func.larmor_freq(1) == Muon.gyro_ratio
    # Test negative handling
    assert func.larmor_freq(-1) == Muon.gyro_ratio
    # Test small numbers
    assert pytest.approx(func.larmor_freq(1e-9)) == 1e-9 * Muon.gyro_ratio


def test_asym():
    print("Testing func.asym...")
    assert func.asym(1, 0) == 1  # Only forwards
    assert func.asym(0, 1) == -1  # Only backwards
    assert func.asym(1, 1) == 0  # Numerator == 0
    assert func.asym(0, 0) == 0  # Denominator == 0
    assert func.asym(-1, 0) == 1  # One negative
    assert func.asym(-1, -1) == 0  # Both negative


def test_decay():
    print("Testing func.decay...")
    pass


def test_Muon():
    print("Testing Muon class...")
    particle = Muon()
    assert particle.charge == 1
    assert pytest.approx(particle.decay_const) == 301764.424861809
    assert pytest.approx(particle.gyro_ratio) == 851371609.122834
    assert particle.mass_energy == 105.6583745e6
    #assert particle.lifetime > 0 and particle.lifetime < 1e-3


def test_inv_decay():
    particle = Muon()
    for U in [0, 0.1, 0.5, 0.9, 1]:
        life = particle.inv_decay(U)
        assert life >= 0
        assert type(life) == np.float64
    with pytest.raises(ValueError):
        assert particle.inv_decay(2)

def test_get_decay_orientation():
    particle = Muon()
    assert particle.get_decay_orientation(1) == particle.gyro_ratio * particle.lifetime

def test_get_asym():
    
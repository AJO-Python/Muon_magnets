#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
import numpy as np
import pytest
from Modules.muon import Muon

def test_Muon():
    muon = Muon()
    print("Testing Muon class...")
    assert muon.charge == 1
    assert muon.mass_energy == 105.6583745e6
    assert muon.halflife == 2.2969811e-6
    assert pytest.approx(muon.gyro_ratio) == 851371609.122834
    assert pytest.approx(muon.decay_const) == 301764.424861809
    assert hasattr(muon, "lifetime")

def test_apply_field():
    muon = Muon()
    muon.apply_field()
    assert hasattr(muon, "larmor")
    assert hasattr(muon, "spin_field_angle")
    assert hasattr(muon, "phase")
    assert hasattr(muon, "polarisation")
    assert hasattr(muon, "total_rads")

def test_inv_decay():
    muon = Muon()
    for U in [0, 0.1, 0.5, 0.9]:
        muon.set_lifetime(U)
        assert muon.lifetime > 0
        assert type(muon.lifetime) == np.float64
    with pytest.raises(ValueError):
        assert muon.set_lifetime(1)
        assert muon.set_lifetime(2)

def test_randomise_phase():
    muon = Muon()
    muon.randomise_phase()
    assert hasattr(muon, "phase")
    assert type(muon.phase) == float

def test_set_larmor():
    muon = Muon()
    muon.set_larmor(1)
    assert muon.larmor == muon.gyro_ratio
    for mag in range(-1, 2, 1):
        muon = Muon()
        muon.set_larmor(mag)
        assert hasattr(muon, "larmor")
        assert muon.larmor >= 0
    
def test_set_spin_polarisation():
    muon = Muon()
    muon.apply_field()
    muon.set_spin_polarisation()
    polar_val = muon.polarisation.item()
    assert hasattr(muon, "polarisation")
    assert (polar_val <= 1 and polar_val >= -1)

def test_set_decay_orientation():
    muon = Muon()
    muon.set_larmor(1)
    muon.set_decay_orientation()
    assert hasattr(muon, "total_rads")
    assert muon.total_rads == muon.gyro_ratio * muon.lifetime + muon.phase

def test_set_asym():
    muon = Muon()
    muon.spin_dir = [0, 1, 0]
    muon.apply_field()
    muon.set_asym(1)
    print(muon.asym)
    assert hasattr(muon, "asym")
    assert (muon.asym >= -1 and muon.asym <= 1)

test_set_asym()


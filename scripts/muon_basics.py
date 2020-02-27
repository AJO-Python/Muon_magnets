#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Anthony J Owen
Muon project
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Modules.functions as func
from Modules.muon import Muon
from Modules.positron import Positron

#%%
# =============================================================================
# Create Muon array
# =============================================================================
muon_array = [Muon() for _ in range(1000)]
lifetimes = [muon.lifetime for muon in muon_array]

plt.figure()
plt.hist(lifetimes, range=[0, 20e-6],
         bins=1000,
         histtype="step",
         cumulative=True)
plt.annotate(s="Target Halflife",
             xy=[2.2e-6, 0], xytext=[2.2e-6, 80000],
             arrowprops={"width":0, "headwidth":0})
plt.annotate(s="Actual Halflife",
             xy=[0, 40000], xytext=[6e-6, 40000],
             arrowprops={"width":0, "headwidth":0})
plt.xlabel("Time of decay ($\mu$s)")
plt.ylabel("Frequency")
plt.ylim(200)
plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
plt.grid()
plt.show()
#%%
# =============================================================================
# Forward and Backward detection of simple precession
# =============================================================================
m1 = Muon()
omega_larmor = func.larmor_freq(1, m1.gamma_u)
forward, backward, for_time, back_time, both = list(), list(), list(), list(), list()
for particle in range(int(2e5)):
    temp_particle = Muon()
    lifetime = temp_particle.lifetime
    P = np.sin(omega_larmor * lifetime)
    if P >= 0:
        forward.append(lifetime)
        for_time.append(lifetime)
    else:
        backward.append(lifetime)
        back_time.append(lifetime)
    both.append(lifetime)

#%%
#plt.figure()
n_f, b_f, _ = plt.hist(forward, histtype="step",
                       bins=1000, label="Forward", range=(0, 20e-6))
n_b, b_b, _ = plt.hist(backward, histtype="step",
                       bins=1000, label="Backward", range=(0, 20e-6))
n_a, b_a, _ = plt.hist(both, histtype="step",
                       bins=1000, label="Combined", range=(0, 20e-6))
plt.title("Histogram")
plt.grid()
#%%
plt.figure()
plt.plot(b_f[:-1], n_f, label="Forward")
plt.plot(b_b[:-1], n_b, label="Backward")
plt.plot(b_a[:-1], n_a, label="Both")
plt.title("Line graph")
plt.xlabel("Lifetime ($\mu$s)")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.grid()
plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
plt.xlabel("Time (s)")
#plt.savefig("Images/lifetime_hist")
#%%
#==============================================================================
# Precession of polarisation
#==============================================================================
line = list(["-", "-.", "--"])
#theta_list = list([0, np.pi/3, np.pi/2, np.pi, np.pi*2/3, np.pi*2])
theta_list = list([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
time_array = np.linspace(0, 100e-6, 1000)
plt.figure()
for i, theta in enumerate(theta_list):
    result = list()
    for t in time_array:
        result.append(Muon().set_spin_polarisation(0.0001, theta))
    plt.plot(result, label="{:.2f}$\pi$".format(theta/np.pi), linestyle=np.random.choice(line), alpha=1)
plt.legend(loc="best")
plt.title("Polarisation as a function of theta and time")
plt.xlabel("Time ({:.1e})".format(max(time_array)))
plt.ylabel("Polarisation ($\sigma$)")
plt.grid()
plt.savefig("Images/Polarisation_theta.png")

#%%
plt.figure()
plt.title("Kubo-Toyabe Relaxation in zero field")
plt.xlabel("Time")
plt.ylabel("Polarisation")
plt.grid()
for angle in [1]:
    polar = list()
    for t in time_array:
        current_P = func.polarisation(m1.decay_const, t)
        polar.append(current_P)
    plt.plot(time_array, polar)
plot_name = "KuboToyabeRelaxation_ZeroField"
plt.xlabel("Time (s)")
plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
plt.savefig("Images/{}".format(plot_name))



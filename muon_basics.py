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
import Modules.muon
import Modules.positron
import Modules.functions
#%%
# =============================================================================
# Main
# =============================================================================

m1 = muon()
omega = larmor_freq(field, m1.gamma_u)

forward, backward, for_time, back_time, both = list(), list(), list(), list(), list()
for particle in range(int(2e5)):
    temp_particle = muon()
    lifetime = temp_particle.life
    #lifetime = inv_decay(np.random.rand())
    P = angular_precession(lifetime, omega, np.pi*2/3)
    if P >= 0:
        forward.append(lifetime)
        for_time.append(lifetime)
    else:
        backward.append(lifetime)
        back_time.append(lifetime)
    both.append(lifetime)

#%%
plt.figure()
n_f, b_f, _ = plt.hist(forward, histtype="step",
                       bins=1000, label="Forward", range=(0, 50))
n_b, b_b, _ = plt.hist(backward, histtype="step",
                       bins=1000, label="Backward", range=(0, 50))
n_a, b_a, _ = plt.hist(both, histtype="step",
                       bins=1000, label="Combined", range=(0, 50))

plt.plot(b_f[:-1], n_f, label="Forward")
plt.plot(b_b[:-1], n_b, label="Backward")
plt.plot(b_a[:-1], n_a, label="Both")
plt.xlim(0, 10)
plt.title("Plot of particles detected against time (N=2e5, theta=2$\pi$3)")
plt.xlabel("Lifetime ($\mu$s)")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.grid()
plt.savefig("Images/lifetime_hist")
#%%
#==============================================================================
# Precession of polarisation
#==============================================================================
line = list(["-", "-.", "--", "--", "-.", "--", "-", "-", "-.", "--", "-", "-.", "--"])
#theta_list = list([0, np.pi/3, np.pi/2, np.pi, np.pi*2/3, np.pi*2])
theta_list = list([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
time_array = np.linspace(0, 100e-6, 1000)
plt.figure()
for i, theta in enumerate(theta_list):
    result = list()
    for t in time_array:
        result.append(angular_precession(t, omega, theta))
    plt.plot(result, label="{:.2f}$\pi$".format(theta/np.pi), linestyle=np.random.choice(line), alpha=1)
plt.legend(loc="best")
plt.title("Polarisation as a function of theta and time")
plt.xlabel("Time ({:.1e})".format(max(time_array)))
plt.ylabel("Polarisation ($\sigma$)")
plt.grid()
plt.savefig("Images/Polarisation_theta")

#%%
plt.figure()
plt.title("Kubo-Toyabe Relaxation in zero field")
plt.xlabel("Time")
plt.ylabel("Polarisation")
plt.grid()
for angle in [1]:
    polar = list()
    for t in time_array:
        current_P = polarisation(m1.decay_const, t)
        polar.append(current_P)
    plt.plot(polar)
plot_name = "KuboToyabeRelaxation_ZeroField"
plt.savefig("Images/{}".format(plot_name))



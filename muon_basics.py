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

# =============================================================================
# Classes
# =============================================================================
class muon:
    def __init__(self):

        self.mass = 1.883531627e-28
        self.charge = 1
        self.mass_energy = 105.6583745e6 # In MeV
        self.halflife = 2296.9811e-9 # In nano seconds e-9
        self.spin = 0.5
        self.gamma_u = 2*np.pi*135.5e6
        self.decay_const = np.log(2)/self.halflife
        self.life = inv_decay(self.decay_const, np.random.rand())
        #self.pos = np.array(position, dtype="float64")
        #self.vel = np.array(velocity, dtype="float64")
        #self.accel = np.array([0, 0, 0], dtype="float64")


class positron:
    def __init__(self):
        self.mass = 9.10938356e-31
        self.charge = 1
        self.mass_energy = 0.5109989461e6
        self.spin = 0.5


# =============================================================================
# Functions
# =============================================================================
def larmor_freq(mag_field, gyro_ratio):
    """
    Returns Larmor frequency
    """
    return get_mag(mag_field) * gyro_ratio


def pos_emit_dir(asym, theta):
    """
    Returns the direction of emitted positron
    """
    return 1 + asym*np.cos(theta)


def asym(Nb, Nf):
    """
    Returns the assymetry of the measurement
    """
    return (Nb - Nf) / (Nb + Nf)


def mag_field(B0, theta):
    """
    Define magnetic fild in a plane X=theta and Y=phi
    B0 in guass
    """
    return [0, B0*np.sin(theta), B0*np.cos(theta)]


def mag_force(q, v, B):
    """
    Returns magnetic force
    F = q ( B X v)
    """
    return q*(np.cross(v, B))


def muon_accel(mu, field):
    """
    Returns muon acceleration under magntic field
    """
    force = mag_force(mu.charge, mu.vel, field)
    accel = force / mu.mass_energy
    return accel


def get_mag(vector):
    """
    Returns magnitude of 3-d vector
    """
    return np.sqrt((vector[0]**2 + vector[1]**2 + vector[2]**2))


def get_gauss(tesla):
    return (tesla/1e4)


def get_unit_vec(vector):
    return vector / get_mag(vector)


def decay(mu, time):
    decay_prob = mu.decay_const * np.exp((-mu.decay_const * time))
    return decay_prob

"""
def decay(decay_const, time):
    decay_prob = 1 - np.exp(-decay_const * time)
    return decay_prob
"""

def inv_decay(decay_const, U):
    """
    Inverse of the decay equation
    Takes a number U={0, 1} and returns decay time
    """
    return -(np.log(U)) / U


def mag_precession(mag_x, w, t):
    return [mag_x*np.cos(w*t), mag_x*np.sin(w*t)]


def angular_precession(t, w, theta):
    return np.cos(theta)**2 + (np.sin(theta)**2)*np.cos(w*t)


def polarisation(decay_const, time):
    """
    Lorentzian Kubo-Toyabe
    """
    lam_t = decay_const * time
    result = (1/3) + ((2/3)*(1-lam_t)*np.exp(-lam_t))
    return result

#%%
# =============================================================================
# Main
# =============================================================================
field = np.array(mag_field(1e-5, 0))

m1 = muon()

omega = larmor_freq(field, m1.gamma_u)
forward, backward, for_time, back_time, both = list(), list(), list(), list(), list()
for particle in range(int(2e6)):
    temp_particle = muon()
    lifetime = temp_particle.life
    #lifetime = inv_decay(m1.decay_const, np.random.rand())
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



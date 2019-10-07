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
    def __init__(self, position, velocity):

        self.mass = 1.883531627e-28
        self.charge = 1
        self.mass_energy = 105.6583745e6
        self.halflife = 2.2969811e-6
        self.spin = 0.5
        self.gamma_u = 2*np.pi*135.5e6
        
        self.pos = np.array(position, dtype="float64")
        self.vel = np.array(velocity, dtype="float64")
        self.accel = np.array([0, 0, 0], dtype="float64")
        

class positron:
    def __init__(self, ):
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
    return mag_field * gyro_ratio


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


def get_magnitude(vector):
    """
    Returns magnitude of 3-d vector
    """
    return np.sqrt((vector[0]**2 + vector[1]**2 + vector[2]**2))


def get_gauss(tesla):
    return (tesla/1e4)


# =============================================================================
# Main
# =============================================================================
field = np.array(mag_field(500000, np.pi/2))

#particles = []
#for i in range(1):
    #rand_pos, rand_vel = np.random.randint(-1000, 1000, size=(2, 3))
    #particles.append(muon(rand_pos, rand_vel))
pos = [2*np.pi, 0, 0]
vel = [0.1, 0, 0.1]
m1 = muon(pos, vel)
m1.accel = muon_accel(m1, field)
dt = 0.5
count = 5000
N = int(count/dt)
m1_pos = np.zeros([N, 3], dtype="float64")
i=0
while i < N:
    m1.accel = muon_accel(m1, field)
    m1.vel += m1.accel*dt
    m1.pos += m1.vel*dt
    m1_pos[i] = m1.pos
    i += 1

# =============================================================================
# Plotting
# =============================================================================
x = m1_pos[0]


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(m1_pos[:,0], m1_pos[:,1], m1_pos[:,2])

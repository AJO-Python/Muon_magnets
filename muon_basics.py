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
        self.mass_energy = 105.6583745e6 # In MeV
        self.halflife = 2296.9811 # In nano seconds e-9
        self.spin = 0.5
        self.gamma_u = 2*np.pi*135.5e6
        self.decay_const = np.log(2)/self.halflife
        
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
    decay_const = np.log(2)/mu.halflife
    decay_prob = decay_const * np.exp((-decay_const * time))
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

def kubo_toyabe(t, w, theta):
    return np.cos(theta)**2 + (np.sin(theta)**2)*np.cos(w*t)
#%%
# =============================================================================
# Main
# =============================================================================
# Creates random particles
#particles = []
#for i in range(1):
    #rand_pos, rand_vel = np.random.randint(-1000, 1000, size=(2, 3))
    #particles.append(muon(rand_pos, rand_vel))

field = np.array(mag_field(1, 0))


# Setting up a standard muon
pos = [2*np.pi, 200, 0.5]
vel = [1.0, 0.0, 1.0]
m1 = muon(pos, vel)
m1.accel = muon_accel(m1, field)

omega = larmor_freq(field, m1.gamma_u)
forward, backward, for_time, back_time, both = list(), list(), list(), list(), list()
for particle in range(int(2e5)):
    lifetime = inv_decay(m1.decay_const, np.random.rand())
    polarisation = kubo_toyabe(lifetime, omega, np.pi*2/3)
    if polarisation >= 0:
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
#plt.yscale("log")
plt.xlim(0, 10)
plt.title("Histogram of particle lifetime (N=2e6, theta=2$\pi$3)")
plt.xlabel("Lifetime (ns)")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.grid()
plt.figure()
plt.plot(b_f[:-1], n_f, label="Forward")
plt.plot(b_b[:-1], n_b, label="Backward")
plt.plot(b_a[:-1], n_a, label="Both")
plt.xlim(0, 10)
plt.title("Plot of particles detected against time (N=2e5, theta=2$\pi$3)")
plt.xlabel("Lifetime (ns)")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.grid()
#%%
#==============================================================================
# Precession of polarisation
#==============================================================================
line = list(["-", "-.", "--", "--", "-.", "--", "-", "-", "-.", "--", "-", "-.", "--"])
theta_list = list([0, np.pi/3, np.pi/2, np.pi, np.pi*2/3, np.pi*2])
#theta_list = list([0, np.pi/3, np.pi/2, np.pi])
plt.figure()
for i, theta in enumerate(theta_list):
    result = list()
    for t in np.linspace(0, 0.2e-13, 1000):
        result.append(kubo_toyabe(t, omega, theta))
    plt.plot(result, label="{:.2f}$\pi$".format(theta/np.pi), linestyle=np.random.choice(line), alpha=1)
plt.legend(loc="best")
plt.title("Polarisation as a function of theta and time")
plt.xlabel("Time (1e-9 s)")
plt.ylabel("Polarisation ($\sigma$)")
plt.grid()
#%%
#==============================================================================
# 
#==============================================================================

# Setting up simulation
dt = 0.1
count = 10000
N = int(count/dt)
m1_pos = np.zeros([N, 3], dtype="float64")
i=0
while i < N:
    m1.vel += 0.5*m1.accel*dt
    m1.pos += m1.vel*dt
    m1.accel = muon_accel(m1, field)
    m1.vel += 0.5*m1.accel*dt
    
    m1_pos[i] = m1.pos
    i += 1

#%%


# =============================================================================
# Plotting
# =============================================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.plot3D(m1_pos[:,0], m1_pos[:,1], m1_pos[:,2],
          label="Muon")


"""
#Plotting the direction of the magnetic field
"""
x_min_max = [min(m1_pos[:,0]), max(m1_pos[:,0])]
y_min_max = [min(m1_pos[:,1]), max(m1_pos[:,1])]
z_min_max = [min(m1_pos[:,2]), max(m1_pos[:,2])]



x_scale = abs(x_min_max[1] - x_min_max[0])
y_scale = abs(y_min_max[1] - y_min_max[0])
z_scale = abs(z_min_max[1] - z_min_max[0])
unit_field = get_unit_vec(field)
ax.plot3D([0, unit_field[0]*0.5*x_scale],
          [0, unit_field[1]*0.5*y_scale],
          [0, unit_field[2]*0.5*z_scale],
          linestyle="--", label="B_field")
ax.legend(loc="lower right")
plt.title("Single Muon moving in B-field with initial velocity in x, z")







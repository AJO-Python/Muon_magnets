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


def get_mag(vector):
    """
    Returns magnitude of 3-d vector
    """
    return np.sqrt((vector[0]**2 + vector[1]**2 + vector[2]**2))


def get_gauss(tesla):
    return (tesla/1e4)


def get_unit_vec(vector):
    return vector / get_mag(vector)


def decay(mu, time, dt):
    decay_const = 1/mu.halflife
    decay_prob = decay_const * np.exp((-decay_const * time))
    return decay_prob*dt

    
#%%
# =============================================================================
# Main
# =============================================================================
field = np.array(mag_field(500000, np.pi/2))

# Creates random particles
#particles = []
#for i in range(1):
    #rand_pos, rand_vel = np.random.randint(-1000, 1000, size=(2, 3))
    #particles.append(muon(rand_pos, rand_vel))

# Setting up a standard muon
pos = [2*np.pi, 0, 0]
vel = [1.0, 0, 1.0]
m1 = muon(pos, vel)
m1.accel = muon_accel(m1, field)

omega = larmor_freq(field, m1.gamma_u)




plt.figure(3)
dt = 4e-6/50
decay_prob = []
x_space = np.linspace(0, 4e-6, 50)
for i in x_space:
    decay_prob.append(decay(m1, i, dt))
    #plt.scatter(i, decay_prob[-1])

#plt.plot(x_space, decay_prob)
n, bins, patches = plt.hist(decay_prob, bins=50, density=True, cumulative=True)
plt.xlim(-1e-6, 5e-6)





# Setting up simulation
dt = 0.05
count = 50
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
fig = plt.figure(2)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.plot3D(m1_pos[:,0], m1_pos[:,1], m1_pos[:,2],
          label="Muon")


"""
Plotting the direction of the magnetic field
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
ax.legend(loc="best")









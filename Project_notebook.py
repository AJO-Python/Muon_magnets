#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import scipy.integrate as integrate

from Modules.muon import Muon
from Modules.positron import Positron
import Modules.functions as func



# Setting graph defaults to avoid repetition on each plot
mpl.rcParams["axes.formatter.limits"] = -2, 2  # Sets xticks to use exponents
mpl.rcParams["axes.grid"] = True  # Turns grid on
mpl.rcParams["legend.loc"] = "best"  # Turns legend on and autoplaces it

# =============================================================================
# Functions
# =============================================================================
def format_subplot(ax, xlab, ylab, title, grid=True, legend=True, show=True):
    ax.set_xlabel(f"{xlab}")
    ax.set_ylabel(f"{ylab}")
    ax.set_title(f"{title}")
    if not legend:
        ax.legend().remove()
    if not grid:
        ax.grid(None)
    if show:
        plt.show()


def set_list_of_values(object_list, attribute):
    """
    attribute.type() = str
    Returns a list of the values for a given attribute of objects in list
    """
    return list(map(lambda x: getattr(x, attribute), object_list))


def theta_predict(W, a0):
    value = (W-1) / a0
    if abs(value) > 1:
        print("np.arccos() can only take {-1, 1}")
        print(f"w: {W})\nValue: {value}")
        return None
    """
    Multiply by 2 because arccos returns {0, pi}
    Desired output should be {0, 2pi}
    """
    return np.arccos((W-1) / a0)*2


def expon(x, A, k):
    """Returns A*e^(kx)"""
    return A*np.exp(x*k)


def set_spin_relaxation(field, time, gyro):
    """
    Returns zero field spin relaxation
    """
    return (1/3) + ((2/3)*(np.cos(gyro * time * field)))


def gaussian_field(width, field):
    """
    Gaussian distribution for muon sensing field
    P_z = y / (sqrt(2pi)*delta) * exp( - (y^2 * B^2) / (2 * delta^2) )
    """
    term1 = (Muon().gyro_ratio) / (((2*np.pi)**0.5) * width)
    term2 = np.exp((-(Muon().gyro_ratio)**2) * (field**2) / (2*(width**2)))
    return term1 * term2


def set_kubo_toyabe_static(width, time):
    """
    Returns static guassian Kubo-Toyabe function
    """
    sigma = Muon().gyro_ratio * width
    t = time
    return (1/3) + (
            (2/3) * (1-((sigma**2) * (t**2))) * np.exp(-0.5*(sigma**2)*(t**2))
            )


def set_field_width(field, axis=None):
    """
    If field is given as a vector, axis must be supplied ["x", "y", "z"]
    Returns gaussian width parameter
    """
    if not axis:
        return Muon().gyro_ratio * abs(field)
    if axis == "x":
        B = field[0]
    elif axis == "y":
        B = field[1]
    elif axis == "z":
        B = field[2]
    return Muon().gyro_ratio * abs(B)


def ext_longitudinal_field(external_field, sigma, omega, time):
    return 1 - (
                2 * (sigma**2) / (omega**2)
            ) * (
                1 - np.cos(omega*time)*np.exp(-0.5*(sigma**2)*(time**2))
            ) + (
                (2 * (sigma**4)/(omega**3)
                ) * (
                    integrate.quad(ext_field_integral, 0, time,
                                   args=(omega, sigma))
                )
            )

    pass
    return polarisation


def ext_field_integral(tau, omega, sigma):
    """Integral function for external_longitudinal_field()"""
    return np.sin(omega * tau) * np.exp(-0.5*(sigma**2)*(tau**2))


def dynamic_field():
    pass
    return some_parameters
#%%
# =============================================================================
#
# =============================================================================

"""
Setting up muons with all attributes.
Generate "N" muons and create/store attributes dependent on
lifetime, field strength, and field direction.
Setting up particles and fields.
"""
N = 10000
field_strength = 3e-3
field_dir = np.array([1, 0, 0])
a0 = 1/3
particles = [Muon() for _ in range(N)]

# Applying field and determining asymmetry
for p in particles:
    """
    Gets larmor_freq, angle between spin and field,
    total_rads revolved and polarisation of Muon()
    """
    p.apply_field(field_dir, field_strength)
    p.asym = func.count_asym(a0, func.larmor_freq(field_strength), p.lifetime)

# Sorting particles list by lifetime
particles.sort(key=lambda x: x.lifetime)

# Storing values in arrays
times = set_list_of_values(particles, "lifetime")
asym = set_list_of_values(particles, "asym")
polar = set_list_of_values(particles, "polarisation")
rads = set_list_of_values(particles, "total_rads")

fields = np.linspace(1e-3, 1e-2, 5)
times =  np.linspace(0, 5e-6, N)

polar_av = np.zeros(N)
polarisation = np.zeros([len(fields), N])
long_polarisation = np.zeros([len(fields), N])
minimum = np.zeros_like(fields)

for i, field in enumerate(fields):
    width = set_field_width(field)
    minimum[i] = np.sqrt(3)/width
    sigma = Muon().gyro_ratio * width
    omega = Muon().gyro_ratio * field
    for j, t in enumerate(times):
        polarisation[i][j] = set_kubo_toyabe_static(width, t)
        long_polarisation[i][j] = ext_longitudinal_field(1e-3, sigma, omega, t)
        
    polar_av = np.add(polar_av, polarisation[i]/len(fields))


plt.plot(times, long_polarisation)
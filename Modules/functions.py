import numpy as np


def larmor_freq(mag_field, gyro_ratio):
    """
    Returns Larmor frequency
    """
    return abs(mag_field * gyro_ratio)


def asym(Nf, Nb):
    """
    Returns the assymetry of the measurement
    """
    Nb = abs(Nb)
    Nf = abs(Nf)
    try:
        return (Nf - Nb) / (Nb + Nf)
    except ZeroDivisionError:
        return 0
def get_mag(vector):
    """
    Returns magnitude of 3-d vector
    """
    return np.sqrt((vector[0]**2 + vector[1]**2 + vector[2]**2))


def mag_force(q, v, B):
    """
    Returns magnetic force
    F = q ( B X v)
    """
    return q*(np.cross(v, B))


def decay(mu, time):
    decay_prob = mu.decay_const * np.exp((-mu.decay_const * time))
    return decay_prob

"""
def decay(decay_const, time):
    decay_prob = 1 - np.exp(-decay_const * time)
    return decay_prob
"""




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

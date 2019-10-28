import Modules

np = Modules.np

gyro_ratio = 2*np.pi*135.5e6
halflife = 2.2969811e-6
decay_const = np.log(2)/halflife

def larmor_freq(mag_field):
    """
    Returns Larmor frequency
    """
    return abs(mag_field * gyro_ratio)


def asym(Nf, Nb):
    """
    Returns the assymetry of the measurement
    """
    if Nf < 0 or Nb < 0:
        print("Negative dectection is not possible. Check func.asym()")
        print("Converting to abs(value) now...")
        Nf = abs(Nf)
        Nb = abs(Nb)
    try:
        return (Nf - Nb) / (Nb + Nf)
    except ZeroDivisionError:
        if Nf == 0 and Nb == 0:
            return 0
        else:
            return 1


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


def decay(time):
    decay_prob = (decay_const * np.exp((-decay_const * time)))
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


def polarisation(time):
    """
    Lorentzian Kubo-Toyabe
    """
    lam_t = decay_const * time
    result = (1/3) + ((2/3)*(1-lam_t)*np.exp(-lam_t))
    return result
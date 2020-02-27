import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate
import Modules.muon as mu
import Modules.dipole as dip

mpl.rcParams["axes.formatter.limits"] = -2, 2  # Sets xticks to use exponents
mpl.rcParams["axes.grid"] = True  # Turns grid on

def static_GKT(time, width):
    """
    :param object time: Length of time to calculate
    :param float width: Guassian width parameter for internal field
    :return: array
    """
    sigma = mu.Muon.gyro_ratio * width
    sig_t_sq = (sigma ** 2) * (time ** 2)
    return ((1 / 3) + ((2 / 3)
                       * (1 - sig_t_sq)
                       * np.exp((-0.5 * sig_t_sq)))
            )


def longitudinal_GKT(time, width, ext_field_in_axis):
    """
    :param object time: Length of time to calculate
    :param float width: Gaussian width parameter for internal field
    :param float ext_field_in_axis: External applied field
    :return: array
    """
    omega = mu.Muon.gyro_ratio * ext_field_in_axis
    sigma = mu.Muon.gyro_ratio * width
    sigma_t = (sigma ** 2) * (time ** 2)

    # Split equation into three parts to ensure correct translation into code
    part_1 = (1 - (
            ((2 * (sigma ** 2)) / (omega ** 2))
            * (1 - (np.cos(omega * time) * np.exp(-0.5 * sigma_t)))
    )
              )

    part_2 = ((2 * (sigma ** 4)) / (omega ** 3))

    part_3 = np.zeros_like(time)
    for i, t in enumerate(time):
        [part_3[i], _] = integrate.quad(longitudinal_integral, a=0, b=t, args=(omega, sigma))
    return part_1 + (part_2 * part_3)


def longitudinal_integral(time, omega, sigma):
    """
    :param float time: Time to integrate at
    :param float omega: larmour precession frequency
    :param float sigma: field width parameter
    :return: float
    """
    sig_time = (sigma ** 2) * (time ** 2)
    return np.sin(omega * time) * np.exp(-0.5 * sig_time)


def static_LKT(time, width):
    """
    :param object time: Length of time to calculate
    :param float width: Lorentzian width parameter for internal field
    :return: array
    """
    a = mu.Muon.gyro_ratio * width
    at = time * a
    return ((1 / 3) - ((2 / 3)
                       * (1 - at)
                       * np.exp((-at)))
            )


def transverse_LKT(time, width, ext_field_in_axis):
    """
    :param object time: Times to calculate over
    :param float width: Lorentzian width parameter for internal field
    :param float ext_field_in_axis: External applied field
    :return: array
    """
    omega = mu.Muon.gyro_ratio * ext_field_in_axis
    sigma = mu.Muon.gyro_ratio * width
    sigma_t = (sigma ** 2) * (time ** 2)


if __name__ == "__main__":
    widths = np.logspace(-4, -3, 5, dtype=float)  # Tesla
    time = np.linspace(0, 20e-6, 150)
    output = np.zeros([len(widths), len(time)])
    av_data = np.zeros(len(time))

    for i, width in enumerate(widths):
        output[i] = longitudinal_GKT(time, width, -1e-3)

    plt.figure()
    for width, data in zip(widths, output):
        plt.plot(time, data, label=f"Width: {width:.1e}T")
        # av_data = np.add(av_data, data)
    # av_data = av_data/len(widths)

    # PLOTTING
    plt.xlabel("Time (s)")
    plt.ylabel("Sum of muon precessions (z-axis)")
    plt.title("Longitudinal (-z) Gaussian Kubo-Toyabe function ($G^{G}_{z}(t)$)")
    plt.legend(loc="best", fontsize="small")
    # plt.plot(time, av_data, "r--", label="Average")
    plt.savefig("Images/Analytical/long_rev_GKT.png", bbox_inches="tight")
    plt.show()

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

import Modules.muon as mu
import Modules.dipole as dip


def gaussian_KT(time, width):
    sigma = mu.Muon.gyro_ratio * width
    sig_t_sq = (sigma ** 2) * (time ** 2)
    return ((1 / 3) + ((2 / 3)
                       * (1 - sig_t_sq)
                       * np.exp((-0.5 * sig_t_sq)))
            )


widths = np.logspace(-4, -3, 10, dtype=float)  # Tesla
time = np.linspace(0, 50e-6, 500)
output = np.zeros([len(widths), len(time)])

for i, width in enumerate(widths):
    output[i] = gaussian_KT(time, width)

plt.figure()
for width, data in zip(widths, output):
    plt.plot(time, data, label=f"{width:.1e}")
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Sum of muon precessions")
plt.title("Static Guassian Kubo-Toyabe function ($G^{G}_{z}(t)$)")
plt.grid()
plt.savefig("Muon_magnets/Images/static_KT.png", bbox_inches="tight")
plt.legend(loc="best", fontsize="small")

plt.show()


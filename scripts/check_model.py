# -*-coding: UTF-8-*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rc_file

rc_file("config/muon.rc")

from modules.island import Island
from modules.muon import Muon
import modules.functions as func

island = Island(orientation=0,
                coord=[0, 0],
                strength=5,
                size=(2, 2),
                location=[0, 0])

muon = Muon(location=[5, 0],
            spin_dir=[-1, 0],
            lifetime=1)

field_calc = island.get_mag_field(muon.location)

"""
By hand calculation for field at (5, 0)m from point dipole at (0, 0)m
with strength 1T, orientation 0 degrees
B = 1.59999e-9 T
"""
print("Field experience by muon is:")
print(f"Hand calculation: [1.599e-9, 0] T")
print(f"Model calculation: {field_calc} T")

########################################################
"""
A muon with spin in the negative x-axis can determine the effect of a dipole
"""
muon.feel_dipole(island)

print("By hand:")
print("angle between spin and field: pi")
print("larmor frequency: 1.362 Hz")
print("angle at decay: 1.362 rads")

print("\nModel values:")
print(str(muon))

#########################################################
"""
Test field drops as expected
"""
locations = np.linspace(0.1, 5, 50)
muons, fields = [], []
for loc in locations:
    temp_muon = Muon(location=[loc, 0])
    temp_muon.feel_dipole(island)
    muons.append(temp_muon)
    fields.append(func.get_mag(temp_muon.field))
fields_normed = [float(i) / max(fields) for i in fields]


def r_3(x, A, c):
    return A * (x ** 3) + c


# popt, pcov = curve_fit(r_3, locations, fields_normed, )
fit_loc = np.linspace(20e-6, 90e-6, 200)

locations_3 = locations ** -0.33
plt.figure()
plt.scatter(locations, fields, label="Muons")
# plt.plot(fit_loc, r_3(fit_loc, *popt), label="Fitted")
plt.legend(loc="best")
plt.ylabel("Field strength")
plt.xlabel("Distance from dipole (m)")
# plt.xlim(20e-6, 90e-6)
plt.ylim(-0.001, 0.0025)
# plt.ylim(0, 1e2)
plt.savefig("images/field_distance.png", bbox_inches="tight")

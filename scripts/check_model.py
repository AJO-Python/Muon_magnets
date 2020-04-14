# -*-coding: UTF-8-*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from modules.dipole import Dipole
from modules.island import Island
from modules.muon import Muon
import modules.functions as func


def check_calculations():
    island = Island(orientation=0,
                    coord=[0, 0],
                    strength=8e-8,
                    size=(0.00002, 0.00002),
                    location=[0, 0])

    muon = Muon(location=[100e-6, 0],
                spin_dir=[-1, 0],
                lifetime=1)

    field_calc = island.get_mag_field(muon.loc)
    hand_calc = 800e3
    print(f"""By hand calculation for field at (100e-6, 0)m from point dipole at (0, 0)m with\n
          strength = 1e-8 T\n
          orientation = 0 deg\n
          B field = {hand_calc} T (NEEDS RECALCULATING)""")
    print(f"Island has magnetic moment {island.moment}")
    print("Field experienced by muon is:")
    print(f"    Hand calculation: {hand_calc} T")
    print(f"    Model calculation: {field_calc} T")

    ########################################################
    print("""\n
    A muon with spin in the negative x-axis can determine the effect of a dipole
    """)
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


def check_muon_dipoles():
    island = Island(orientation=0,
                    coord=[0, 0, 0],
                    strength=1e-8,
                    size=(0.00002, 0.00002),
                    location=[0, 0, 0])
    num_muons = 80
    locations = np.linspace(40e-6, 600e-6, num_muons)
    fields = np.zeros(num_muons, dtype=float)
    muons = np.zeros(num_muons, dtype=object)
    for i, loc in enumerate(locations):
        temp_muon = Muon(location=[loc, 0, 100e-9])
        temp_muon.feel_dipole(island)
        muons[i] = temp_muon
        fields[i] = func.get_mag(temp_muon.field)
    fields = np.array(fields)
    plt.figure()
    plt.scatter(locations, fields, label="Muons")
    plt.legend(loc="best")
    plt.title("3d muon-dipole interaction (z=400 um")
    plt.ylabel("Field strength")
    plt.xlabel("Distance from dipole (m)")
    plt.xlim(func.get_limits(locations))
    plt.ylim(func.get_limits(fields))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.grid()
    plt.savefig("images/field_distance.png", bbox_inches="tight")


###########################################################
def check_precession():
    field = 1e-8
    gyro = 2 * np.pi * 136e6
    larmour = gyro * field
    lifes = np.linspace(0, 20e-6, 100)

    fig, ax = plt.subplots()
    ax.plot(lifes, polarisation_func(lifes, larmour), label="precession")
    ax.legend(loc="best")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Polarisation")
    ax.set_title("Polarisation against time")
    fig, ax = func.make_fancy_plot(fig, ax)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-5, -5))
    ax.grid()
    plt.savefig("images/polarisation_test.png", bbox_inches="tight")


def polarisation_func(lifetime, larmour):
    theta = np.pi / 4
    w = larmour
    t = lifetime
    return np.cos(theta) ** 2 + (np.sin(theta) ** 2) * np.cos(w * t)


####################################
# Plot 1d field for dipole

def check_1d_dipole(norm=False):
    """
    Plots and checks that the dipole in 1D works as expected
    """
    # Set up field test
    source = 0
    targets = np.linspace(20e-6, 100e-6, 20, endpoint=True)
    forces, indices_to_remove = mag_field_1d(source, targets)
    if norm:
        forces = func.normalise(forces)
    # targets = np.delete(targets, indices_to_remove)
    # Curve fit
    popt, pcov = curve_fit(cubic, targets, forces)
    fit_targets = np.linspace(min(targets), max(targets), 100)
    fit_forces = cubic(fit_targets, *popt)
    # PLOT
    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    fig, ax = func.make_fancy_plot(fig, ax)
    ax.scatter(targets, forces, label="Calculated field", marker="x", color="k")
    ax.plot(fit_targets, fit_forces, label="$k/r^3$ fit")
    ax.set_ylim(func.get_limits(fit_forces))
    # ax.set_ylim(0, 1)
    ax.set_xlim(func.get_limits(targets))
    ax.legend(loc="best")
    ax.set_xlabel("Distance from dipole (m)")
    ax.set_ylabel("Normalised field strength B(r) (arb. units)")
    ax.set_title("Dipole field strength against separation (1 dimension)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid()
    filename = "images/1d_field_norm.png" if norm else "images/1d_field.png"
    print(f"File saved as: {filename}")
    print(f"Field at 100e-6: {forces[-1]}")
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def mag_field_1d(source, target):
    """
    :param coordinate source: Location of field source
    :param array(cooridnates) target: the locations to calculate the field at
    :rtype: array
    :return: An array of field magnitudes
    """
    moment = 8e-8
    mag_perm = 1e-7  # Cancel constant terms to get mag_perm as only constant
    relative_loc = target - source
    magnitude = np.sqrt((relative_loc ** 2))
    too_close_locations = np.where(abs(relative_loc) <= float(1e-9))
    relative_loc = np.delete(relative_loc, too_close_locations)
    fields = np.array(mag_perm * (
            (3 * relative_loc * (np.dot(moment, relative_loc))
             / (magnitude ** 5))
            - (moment / (magnitude ** 3))
    ))
    # Simplified field calculation for debugging in 1d
    # fields = np.array(mag_perm * (
    #     2*moment / (relative_loc**3)
    # ))
    # fields[too_close_locations] = max(fields)
    return fields, too_close_locations


def cubic(x, m):
    return 2 * m / (x ** 3)


def check_relaxation():

    island = Island(orientation=0,
                    coord=[0, 0],
                    strength=1e-8,
                    size=(0.00002, 0.00002),
                    location=[0, 0])

    muon = Muon(location=[100e-6, 0],
                spin_dir=[-1, 0],
                lifetime=1)


angles = np.random.uniform(0, 6.28, 10)
ensemble = [Dipole(angle, [0, 0, 0], 3) for i, angle in enumerate(angles)]

if __name__ == "__main__":
    check_calculations()
    check_muon_dipoles()
    check_precession()
    check_1d_dipole(norm=False)

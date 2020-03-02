import numpy as np
import matplotlib.pyplot as plt

from modules.muon import Muon
from modules.dipole_grid import make_dipole_grid
from modules.functions import load_run, get_mag


def make_uniform_muons(N, max_loc):
    """
    :param int N: Number of muons
    :param tuple max_loc: Maximum position value
    :rtype: array
    :return: array of uniformly distributed muons
    """
    # Create uniform random coordinates within grid limits
    rand_loc_x = np.random.uniform(0, max_loc[0], N)
    rand_loc_y = np.random.uniform(0, max_loc[1], N)
    rand_loc = np.column_stack([rand_loc_x, rand_loc_y])
    # Init muon array
    muons = np.empty(N, dtype=object)
    # Spread muons over grid
    for i in range(N):
        muons[i] = Muon(location=rand_loc[i], field=[0, 0])
    return muons


def set_muon_field(dipoles, muons):
    """

    :param array dipoles: array of dipoles
    :param array muons: array of muons
    :return: muons set with their total field
    """
    for i, dipole in enumerate(dipoles):
        print(f"On dipole {i}/{len(dipoles) - 1}")
        for muon in muons:
            # Checks if muon is inside a nano-island
            if (
                    ((muon.location[0] <= dipole.location[0] + 350e-9)
                     and (muon.location[0] >= dipole.location[0] - 350e-9))
                    or ((muon.location[1] >= dipole.location[1] + 8e-7)
                        and (muon.location[1] <= dipole.location[1] - 8e-7))
            ):
                # TODO Add behaviour for inside nano-island
                continue
            muon.field += dipole.get_mag_field(muon.location)


# Make dipole grid
run_name = make_dipole_grid()

# Load dipole grid
dipole_data = load_run(run_name, files=["dipoles"])[0]
dipole_array = dipole_data["dipoles"]
max_loc = dipole_array[-1].location

# Create and place muons
N = 10_000
muons = make_uniform_muons(N, max_loc)
# Calculate field for each muon
set_muon_field(dipole_array, muons)

field_strengths = np.zeros_like(muons)
for i, muon in enumerate(muons):
    field_strengths[i] = get_mag(muon.field)

plt.figure()

plt.hist(field_strengths, bins=100, range=(0, 2e9))

plt.show()

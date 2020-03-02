import numpy as np
import matplotlib.pyplot as plt

from modules.muon import Muon
from modules.dipole_grid import make_dipole_grid
from modules.functions import load_run, get_mag, save_array
from modules.field_grid import calc_field_grid

def spread_muons(run_name, N, dipole_array, max_loc, spread=""):
    """
    Spreads N muons over the dipole grid and sets their attributes
    to hold the sum of fields at the muons location
    Saves the muon array to the run_name directory as "muons"

    :param int N: Number of muons to model
    :param array dipole_array: Dipoles spread over an area
    :param tuple max_loc: Bounding region for muons
    :return: saves array of muons with fields calculated from position on dipole grid
    """
    # Create and place muons
    if spread.lower() == "unifom":
        particles = {"muons": make_uniform_muons(N, max_loc)}
    else:
        particles = {"muons": make_normal_muons(N, max_loc)}

    # Calculate field for each muon
    set_muon_field(dipole_array, particles["muons"])
    save_array(run_name, "muons", **particles)
    return

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


def make_normal_muons(N, max_loc):
    """
    :param int N: Number of muons
    :param tuple max_loc: Maximum position value
    :rtype: array
    :return: array of uniformly distributed muons
    """
    # Create uniform random coordinates within grid limits
    rand_loc_x = np.random.normal(max_loc[0]/2, max_loc[0]/5, N)
    rand_loc_y = np.random.normal(max_loc[1]/2, max_loc[1]/5, N)
    rand_loc = np.column_stack([rand_loc_x, rand_loc_y])
    # Init muon array
    muons = np.empty(N, dtype=object)
    # Spread muons over grid
    for i in range(N):
        muons[i] = Muon(location=rand_loc[i], field=[0, 0])
    return muons


def set_muon_field(dipoles, muons):
    """
    Calculates field experienced by each muon as a sum of contributions
    from all the dipoles in the model

    :param array dipoles: array of dipoles
    :param array muons: array of muons
    :return: sets muon attributes to their total field
    """
    for i, dipole in enumerate(dipoles):
        print(f"On dipole {i}/{len(dipoles) - 1}")
        for muon in muons:
            # Checks if muon is inside a nano-island
            if is_inside_island(dipole, muon):
                # TODO Add behaviour for inside nano-island
                #muon.field += dipole.get_internal_field(muon.location)
                muon.in_island = True
            else:
                muon.field += dipole.get_mag_field(muon.location)
    return

def is_inside_island(dipole, muon):
    """
    :param object dipole: Dipole to check
    :param object muon: Muon to check is inside dipole
    :rtype: bool
    :return: True if muon is inside dipole
    """
    # Check if muon is inside x_coords
    if ( (muon.location[0] <= dipole.location[0] + 350e-9)
        and (muon.location[0] >= dipole.location[0] - 350e-9)):
        inside_x = True
    else:
        return False  # Early exit condition

    # Check if muon is inside y_coords
    if ( (muon.location[1] <= dipole.location[1] + 8e-7)
        and (muon.location[1] >= dipole.location[1] - 8e-7)):
        return inside_x
    else:
        return False


# =============================================================================
#
# =============================================================================
if __name__=="__main__":
    """
    Create dipole grid and scatter muons over it
    Gets field widths and values
    """
    N = 10_000

    # Make dipole grid
    #run_name = make_dipole_grid()
    run_name = "5x5_U_1"
    # Load dipole grid
    dipole_data = load_run(run_name, files=["dipoles"])
    dipole_array = dipole_data["dipoles"]["dipoles"]
    max_loc = dipole_array[-1].location

    # Spread muons over dipoles and calculate field values
    spread_muons(run_name=run_name, N=N, dipole_array=dipole_array, max_loc=max_loc)
    muons_data = load_run(run_name, files=["muons"])
    muons = muons_data["muons"]["muons"]
    field_strengths = {"total": np.zeros_like(muons),
                       "x_axis": np.zeros_like(muons),
                       "y_axis": np.zeros_like(muons)}
    for i, muon in enumerate(muons):
        field = muon.field
        field_strengths["total"][i] = get_mag(field)
        field_strengths["x_axis"][i] = field[0]
        field_strengths["y_axis"][i] = field[1]


    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.hist(field_strengths["total"], bins=100, label="field magnitude")
    ax2.hist(field_strengths["x_axis"], bins=100, label="x field")
    ax3.hist(field_strengths["y_axis"], bins=100, label="y field")
    ax1.legend(loc="best")
    fig.show()

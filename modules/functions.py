import numpy as np
import matplotlib.pyplot as plt

from modules.grid import Grid
from modules.ensemble import Ensemble


# =============================================================================
# VECTOR FUNCTIONS
# =============================================================================
def get_mag(vector):
    """
    :param array vector: Vector to operate on
    :rtype: float
    :return: Magnitude of $vector
    """
    vector = np.array(vector)
    return np.sqrt(np.dot(vector, vector))


def get_unit_vector(vector):
    """
    :param array vector: Vector to operate on
    :rtype: array
    :return: unit vector of $vector
    """
    norm = np.linalg.norm(vector)
    if not norm:
        return vector
    else:
        return vector / norm


def get_angle(vec1, vec2):
    """
    :param array vec1: First vector
    :param array vec2: Second vector
    :rtype: float
    :return: Angle between $vec1 and $vec2 in radians {-pi/2, pi/2}
    """
    unit1 = get_unit_vector(vec1)
    unit2 = get_unit_vector(vec2)
    # dot = sum(x * y for x, y in zip(vec1, vec2))
    # mag1, mag2 = get_mag(vec1), get_mag(vec2)
    return np.arccos(np.clip(np.dot(unit1, unit2), -1, 1))


def normalise(arr):
    """
    :param array arr: Input array to normalise
    :rtype: array
    :return: Normalised array
    """
    norm = np.sqrt((arr ** 2).sum())
    if norm == 0:
        return arr
    return arr / norm


# =============================================================================
# Plotting
# =============================================================================
def get_limits(arr):
    """
    :param array arr: Data to get plotting limits for
    :rtype: tuple(float, float)
    :return: min and max for plotting arr
    """
    spread = max(arr) - min(arr)
    if spread == 0:
        spread = max(arr)
    small = min(arr) - spread / 20
    big = max(arr) + spread / 20
    return (small, big)


def make_fancy_plot(fig, ax):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
    plt.style.use("./config/fancy_plots.mplstyle")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    return fig, ax


def set_fig_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    :param float width: Document textwidth or columnwidth in pts (Set for muon report by default)
    :param float fraction: optional, Fraction of the width which you wish the figure to occupy
    :param tuple (int, int) subplots: Subplot dimensions of the figure
    :returns tuple fig_dim: Dimensions of figure in inches
    """
    if width == "muon_paper":
        width = 426.79134
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


# =============================================================================
# Multiple runs
# =============================================================================
def setup_run_save(dipole_configs, muon_configs, num_muons=20000):
    """
    Sets up and saves results for runs determined by input config files
    :param list (str) dipole_configs: List of configuration files for dipoles grid
    :param list (str) muon_configs: list of config files for ensemble
    :param int num_muons: Number of muons in ensemble
    :return: None
    """
    for d_file, m_file in zip(dipole_configs, muon_configs):
        island_grid = Grid(config_file=d_file)
        RUN_NAME = island_grid.run_name
        particles = Ensemble(N=num_muons,
                             run_name=RUN_NAME,
                             config_file=m_file)
        # particles.set_generic("spin_dir", [1, 0, 0])
        particles.calculate_fields(island_grid)
        particles.load_fields()
        particles.set_relaxations()
        particles.save_ensemble()
        particles.plot_relax_fields(save=True)
        particles.plot_distribution(grid=island_grid, save=True)
# =============================================================================
# DATA SAVE
# =============================================================================
def save_array(run_name, file_name, **kwargs):
    """
    Saves arrays to "Muon_magnets/data/{run_name}/{file_name}.npz"

    :param str run_name: Name of run
    :param str file_name: Name of file
    :param dict kwargs: {"data_name": data} arrays to save to file
    :return: Saves arrays to a binary .npy file
    """
    import os
    import errno
    # Check if run folder exists and make it if not
    try:
        os.makedirs(f"data/{run_name}")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    file_path = f"data/{run_name}/{file_name}.npz"
    if os.path.exists(file_path):
        print("Inside delete block")
        os.remove(file_path)
    np.savez(file_path, **kwargs)
    print(f"Saved to {file_path}")


def save_object(run_name, file_name, obj):
    """
    :param str filename: Will save to "Muon_magnets/data/{run_name}/{filename}.pickle"
    :param object obj: Object to save
    """
    import pickle
    file_path = f"data/{run_name}/{file_name}.pickle"
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        print(f"Saving pickle to {file_path}")
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# =============================================================================
# DATA LOAD
# =============================================================================
def load_object(run_name, file_name):
    """
    :param str filename: Will load from "Muon_magnets/data/{filename}.pickle"
    :rtype: object
    :return: Object stored in file
    """
    import pickle
    file_path = f"data/{run_name}/{file_name}.pickle"
    with open(file_path, 'rb') as output:  # Open as read
        print(f"Loading pickle from {file_path}")
        return pickle.load(output)


def load_config(file_name):
    """
    All variables must be on a newline and seperated from its value by "="
    :param str file_name: Name of file in Muon_magnets/config/{file_name}
    :rtype: dict
    :return: Dictionary of config variables
    """
    # Load file as all string
    load_data = np.loadtxt(f"config/{file_name}.txt",
                           delimiter="\n",
                           dtype=str)
    # Unpack into a dictionary and convert to float/bool
    data = {}
    for item in load_data:
        key, value = item.split("=")
        try:
            data[key.strip()] = float(value.strip())
        except ValueError:
            # print(f"\"{item}\" is not float. Converting to bool...")
            data[key.strip()] = True if value.strip().lower() == "true" else False
            # print(f"Converted {key.strip()} to bool -> {data[key.strip()]}")
    return data


def load_run(run_name, files=[]):
    """
    :param str run_name: Folder run is saved to
    :rtype: Dict
    :return: Three dictionaries with dipole, field, and location data
    """
    data = {}
    if files:
        for i, file in enumerate(files):
            data[file] = np.load(f"data/{run_name}/{file}.npz", allow_pickle=True)
            print(f"Loaded {file}...")
        return data

    else:
        dipole_data = np.load(f"data/{run_name}/dipoles.npz", allow_pickle=True)
        print(f"Loaded dipoles")

        field_data = np.load(f"data/{run_name}/fields.npz")
        print(f"Loaded fields")

        loc_data = np.load(f"data/{run_name}/locations.npz")
        print(f"Loaded locations")

        return dipole_data, field_data, loc_data

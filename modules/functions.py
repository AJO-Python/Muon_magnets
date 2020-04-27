import numpy as np
import matplotlib.pyplot as plt

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


# =============================================================================
# PARTICLE FUNCTIONS
# =============================================================================
def detect_asym(Nf, Nb):
    """
    :param int Nf: Number of counts in forward detector
    :param int Nb: Number of counts in backward detector
    :rtype: float
    :return: Asymmetry
    """
    # Catching and fixing negative detection
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


def format_plot(fig, max_time=20e-6):
    fig.legend(loc="best")
    fig.xlim(0, max_time)
    fig.grid()
    fig.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
    fig.show()


def chunk_muons(list_to_chunk, freq_per_chunk):
    """
    :param list list_to_chunk: A sorted list of objects to split into chunks
    :param list freq_per_chunk: A list of counts per bin for $list_to_chunk
    :rtype: array
    :return: Array made from $list_to_chunk split into "freq_per_chunk" length chunks
    Creates list of specific muons decaying for each time step
    """
    chunks = np.zeros_like(freq_per_chunk, dtype="object")
    chunk_start = 0
    for i, freq in enumerate(freq_per_chunk):
        freq = int(freq)
        chunks[i] = list_to_chunk[chunk_start:chunk_start + freq]
        chunk_start += freq
    return chunks


def plot_relaxations(particles, RUN_NAME, GRAPH_NAME, field_dict):
    from scipy.optimize import curve_fit
    from modules.model_equations import static_GKT
    from modules.muon import Muon

    overall = np.nansum(particles.relaxations, axis=0) / particles.N

    # CURVE FIT
    popt, pcov = curve_fit(static_GKT, Muon.TIME_ARRAY, overall, p0=1e-4)

    # Setup subplots
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax4 = plt.subplot2grid((2, 3), (1, 2))

    field_axes = (ax1, ax2, ax3, ax4)
    # Plot individual lines if N is small
    if len(particles.relaxations) < 100:
        for i in range(N):
            ax0.plot(Muon.TIME_ARRAY, particles.relaxations[i], alpha=0.5, lw=0.5)

    # Plot overall relaxation
    ax0.plot(Muon.TIME_ARRAY, overall, lw=2, c="k", alpha=0.7, label="Model")
    ax0.plot(Muon.TIME_ARRAY, static_GKT(Muon.TIME_ARRAY, *popt), c="r", label="Curve fit")

    ax0.legend(loc="upper right")
    ax0.set_xlim(0, Muon.TIME_ARRAY[-1])
    ax0.set_xlim(0, 20e-6)
    ax0.grid()
    ax0.set_title("Relaxation function from dipole grid")
    ax0.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))

    ax1.set_title("Magnitudes of overall field")

    for sub_ax, field in zip(field_axes, field_dict.keys()):
        sub_ax.hist(field_dict[field], bins=100)
        sub_ax.set_title(f"Magnitudes of {field}")
        sub_ax.set_xlabel("Field strength (T)")
        sub_ax.set_ylabel("Frequency")
        sub_ax.grid()
        sub_ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
    # Add legend
    plt.tight_layout(pad=1)
    plt.savefig(f"data/{RUN_NAME}/{GRAPH_NAME}.png")
    # print(f"Actual width: {random_width}")
    print(f"Calculated width: {popt} +- {pcov}")
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
    np.savez(file_path, **kwargs)
    print(f"Saved to {file_path}")


def save_object(run_name, file_name, obj):
    """
    :param str filename: Will save to "Muon_magnets/data/{filename}.txt"
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


def load_fields(run_name):
    """
    Loads field experiences by each muon from multiprocessing calculation
    :param str run_name: run folder
    :rtype: array, dict
    :return: Array of fields and Dictionary contating mag, x, y, z fields for each muon
    """
    fields = load_run(run_name, files=["muon_fields"])
    fields = np.array(fields["muon_fields"]["muon_fields"])

    magnitudes = np.array([get_mag(f) for f in fields])
    field_dict = {"total": magnitudes, "x": fields[:, 0], "y": fields[:, 1], "z": fields[:, 2]}
    return fields, field_dict

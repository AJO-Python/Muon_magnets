import numpy as np
"""
VECTOR FUNCTIONS
"""
def get_mag(vector):
    """
    :param array vector: Vector to operate on
    :rtype: float
    :return: Mangitude of $vector
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
    :return: Angle between $vec1 and $vec2 in radians {-pi/2, pi/2
    """
    dot = sum(x * y for x, y in zip(vec1, vec2))
    mag1, mag2 = get_mag(vec1), get_mag(vec2)
    return np.arccos(dot / (mag1 * mag2))

"""
PARTICLE FUNCTIONS
"""

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

"""
DATA SAVE/LOAD FUNCTIONS
"""
def save_array(filename, **kwargs):
    """
    :param str filename: Will save to "Muon_magnets/data/{filename}.txt"
    :param array data: 2d array to save to file
    :return: Saves an array to a binary .npy file
    """
    file_path = f"./data/{filename}.npz"
    #print(kwargs)
    # TODO: Work out why **kwargs is not saving files properly
    # Currently stores as "arr_01"
    # It does not unpack the args so they are not being labelled correctly
    np.savez(file_path, )
    print(f"Saved to {file_path}")


def save_object(filename, obj):
    """
    :param filename: Will save to "Muon_magnets/data/{filename}.txt"
    :param obj: Object to save
    """
    import pickle
    file_path = f"./data/{filename}.pickle"
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    :param filename: Will load from "Muon_magnets/data/{filename}.pickle"
    :rtype: object
    :return: Object stored in file
    """
    import pickle
    file_path = f"./data/{filename}.pickle"
    with open(file_path, 'rb') as output:  # Overwrites any existing file.
        return pickle.load(output)

import numpy as np

gyro_ratio = 2*np.pi*135.5e6  # Radians per second per Tesla (rad s^-1 T^-1)
halflife = 2.2969811e-6  # Seconds
decay_const = np.log(2)/halflife  # Seconds
"""
VECTOR FUNCTIONS
"""
def get_mag(vector):
    """
    Returns magnitude of 3-d vector
    """
    vector = np.array(vector)
    return np.sqrt(np.dot(vector, vector))


def get_unit_vector(vector):
    norm = np.linalg.norm(vector)
    if not norm:
        return vector
    else:
        return vector/norm


def get_angle(vec1, vec2):
    """
    Gets angle between vectors of shape (N,1) as theta {-pi2, pi/2}
    """
    dot = sum(x*y for x,y in zip(vec1, vec2))
    mag1, mag2 = get_mag(vec1), get_mag(vec2)
    return np.arccos(dot / (mag1*mag2))

"""
PARTICLE FUNCTIONS
"""
def detect_asym(Nf, Nb):
    """
    Returns the asymmetry of the measurement
    """
    # Catching and reporting negative detection
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
    Creates list of specific muons decaying for each time step
    freq_per_chunk comes from a histogram of muon decay times
    list_to_chunk -> sorted list of objects to chunk
    """
    chunks = np.zeros_like(freq_per_chunk, dtype="object")
    chunk_start = 0
    for i, freq in enumerate(freq_per_chunk):
        freq = int(freq)
        chunks[i] = list_to_chunk[chunk_start:chunk_start+freq]
        chunk_start += freq
    return chunks

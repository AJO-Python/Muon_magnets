import numpy
class Positron:
    def __init__(self):
        self.mass = 9.10938356e-31
        self.charge = 1
        self.mass_energy = 0.5109989461e6
        self.spin = 0.5

    def emit_dir(asym, theta):
        """
        Returns the direction of emitted positron
        """
        return 1 + asym*np.cos(theta)

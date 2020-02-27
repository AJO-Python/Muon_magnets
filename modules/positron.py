from Modules import *
class Positron:
    def __init__(self):
        self.mass = 9.10938356e-31
        self.charge = 1
        self.mass_energy = 0.5109989461e6
        self.spin = 0.5

    def emit_dir(a0, theta):
        """
        Returns the direction of emitted positron
        """
        return 1 + a0*np.cos(theta)

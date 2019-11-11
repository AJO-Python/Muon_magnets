import numpy as np
import Modules.functions as func

class Muon:
    mass = 1.883531627e-28
    charge = 1
    mass_energy = 105.6583745e6
    halflife = 2.2969811e-6
    gyro_ratio = 2*np.pi*135.5e6
    decay_const = np.log(2)/halflife
    spin_dir = np.array([0, 0, -1])
    phase = 0

    def __init__(self):
        """
        All values for muons in SI units
        """
        # Setting each muon to have unique lifetime based on exponential decay function
        self.lifetime = float(self.inv_decay(np.random.rand(1)))
       
    def apply_field(self, field_dir=[1, 0, 0], field_strength=1e-3):
        """
        Must set Muon.phase before calling apply_field
        Sets muon attributes if field is present
        field_dir must be x,y,z vector
        Defaults to: 1mT field in +x direction + 0 phase
        """
        self.get_larmor(field_strength)
        self.spin_field_angle = func.get_angle(self.spin_dir, field_dir)
        self.get_decay_orientation(phase=0)
        self.get_spin_polarisation()
        
    def inv_decay(self, U):
        """
        Inverse of the decay equation
        Takes a number U={0, 1} and returns decay time
        """
        if U > 1:
            raise ValueError("U must be in range {0, 1}")
        if U == 0:  # Stopping inf from taking log of 0
            U = 1e-9
        return -(np.log(U)) / Muon.decay_const

    def get_random_phase(self):
        """Sets phase to random angle"""
        self.phase = np.rand.uniform(0, 2*np.pi)

    def get_larmor(self, field_mag):
        """Returns omega_l in radians"""
        self.larmor = field_mag * self.gyro_ratio

    def get_spin_polarisation(self):
        """
        Returns polarisation as function of field strength
        and angle from muon spin dir
        """
        theta = self.spin_field_angle
        w = self.larmor
        t = self.lifetime
        p = self.phase
        self.polarisation = np.cos(theta)**2 +
            (np.sin(theta)**2) *
            np.cos(w*t+p)

    def get_decay_orientation(self):
        """Return orientation at decay as total radians"""
        self.total_rads = self.larmor*self.lifetime + self.phase

    def get_asym(self, a0, larmor):
        """
        Returns asymmetry
        """
        if a0 <= -1 or a0 >= 1:
            print("a0 out of range {-1, 1}")
            return 0
        return a0 * np.cos(self.larmor) * self.lifetime

import numpy as np
if __name__ == "__main__":
    import functions as func
else:
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
        Initialises each muon object with a lifetime
        """
        self.get_lifetime(np.random.rand(1))
       
    def apply_field(self, field_dir=[1, 0, 0], field_strength=1e-3, random_phase=False):
        """
        Must set Muon.phase before calling apply_field
        Sets muon attributes if field is present
        field_dir must be x,y,z vector
        Defaults to: 1mT field in +x direction + 0 phase
        """
        try:
            if random_phase:
                self.get_random_phase()
            self.get_spin_field_angle(field_dir)
            self.get_larmor(field_strength)
            self.get_decay_orientation()
            self.get_spin_polarisation()
        except:
            print("Unexpected error in apply_field()")
            raise

    def get_lifetime(self, U):
        """
        Returns decay time in s
        Inverse of the decay equation
        Takes a number U={0, 1} and returns decay time
        """
        if U >= 1:
            raise ValueError("U must be in range {0, 1}")
        if U == 0:  # ln(0) -> inf so set to some small value
            U = 1e-9
        self.lifetime = -(np.log(U)) / Muon.decay_const

    def get_random_phase(self):
        """Sets phase to random radian"""
        self.phase = np.random.uniform(0, 2*np.pi)

    def get_larmor(self, field_mag):
        """Sets larmor to radians per second"""
        self.larmor = abs(field_mag) * self.gyro_ratio

    def get_spin_polarisation(self):
        """
        Sets polarisation as function of field strength
        and angle from muon spin direction
        """
        try:
            theta = self.spin_field_angle
            w = self.larmor
            t = self.lifetime
            p = self.phase
        except AttributeError as e:
            print(f"{e}: must be defined before calling get_spin_polarisation")
            return
        self.polarisation = (np.cos(theta)**2 +
                             (np.sin(theta)**2) * np.cos(w*t+p))

    def get_decay_orientation(self):
        """Sets orientation at decay as total radians"""
        try:
            self.total_rads = self.larmor * self.lifetime + self.phase
        except AttributeError as e:
            print(f"{e}: must be defined before calling get_spin_polarisation")
            return

    def get_asym(self, a0):
        """
        Sets asymmetry
        """
        if a0 < -1 or a0 > 1:  # Checking a0 is in correct range
            print("a0 out of range {-1, 1}")
            return 0
        # Checking object has spin_field attribute
        try:
            asym = 1 + a0 * np.cos(self.spin_field_angle * self.lifetime)
            self.asym = asym
        except AttributeError as e:
            print(f"{e}: must be defined before calling get_asym")
            return

    def get_spin_field_angle(self, field_dir):
        """Sets spin_field_angle in radians"""
        self.spin_field_angle = func.get_angle(self.spin_dir, field_dir)

    def get_kubo_toyabe(self, width):
        t = self.lifetime
        self.kt = (1/3) + ( (2/3) * (1-((width**2) * (t**2))) *
                   np.exp(-0.5*(width**2)*(t**2)) )
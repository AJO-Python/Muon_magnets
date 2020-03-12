import numpy as np
import modules.functions as func


class Muon:
    """
    Makes an object representing a muon decaying at a set location in a magnetic field
    """
    MASS = 1.883531627e-28
    charge = 1
    mass_energy = 105.6583745e6
    halflife = 2.2969811e-6
    gyro_ratio = 2 * np.pi * 135.5e6
    decay_const = np.log(2) / halflife
    spin_dir = np.array([0, 0, -1])
    phase = 0

    def __init__(self, **kwargs):
        """
        Initialises each muon object with a lifetime and optional kwargs

        :param dict kwargs: Additional properties and values to give to muon
        """
        self.set_lifetime(np.random.rand(1))  # Set this before kwargs so it can be overwritten if needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """
        :rtpe: str
        :return: list of all attributes and values for self
        """
        return "\n".join([": ".join((str(att), str(getattr(self, att)))) for att in vars(self)])

    def __repr__(self):
        return f"Muon at {self.location}"

    def apply_field(self, field_dir=[1, 0, 0], field_strength=1e-3, random_phase=False):
        """
        Must set Muon.phase before calling apply_field
        Sets muon attributes if field is present

        :param array field_dir: (x,y,z) unit vector for field
        :param float field_strength: Strength of field at muon
        :param bool random_phase: Choose to randomise muon phase
        :rtype: setter
        :return: Sets
        """
        try:
            if random_phase:
                self.randomise_phase()
            self.set_spin_field_angle(field_dir)
            self.set_larmor(field_strength)
            self.set_decay_orientation()
            self.set_spin_polarisation()
        except:
            print("Unexpected error in apply_field()")
            raise

    def set_lifetime(self, U):
        """
        Inverse of the decay equation
        Takes a number U={0, 1}
        Returns decay time in seconds
        """
        if U >= 1:
            raise ValueError("U must be in range {0, 1}")
        if U == 0:  # ln(0) -> inf so set to some small value
            U = 1e-9
        self.lifetime = -(np.log(U)) / Muon.decay_const

    def randomise_phase(self):
        """Sets phase to random radian"""
        self.phase = np.random.uniform(0, 2 * np.pi)

    def set_larmor(self, field_mag):
        """Sets larmor to radians per second"""
        self.larmor = abs(field_mag) * self.gyro_ratio

    def set_spin_polarisation(self):
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
            print(f"{e}: must be defined before calling set_spin_polarisation")
            return
        self.polarisation = (np.cos(theta) ** 2 +
                             (np.sin(theta) ** 2) * np.cos(w * t + p))

    def set_decay_orientation(self):
        """Sets orientation at decay as total radians"""
        try:
            self.total_rads = (self.larmor * self.lifetime + self.phase) % (2 * np.pi)
        except AttributeError as e:
            print(f"{e}: must be defined before calling set_spin_polarisation")
            raise

    def set_asym(self, a0):
        """
        Sets asymmetry
        a0 is bulk asymmetry of muon ensemble
        """
        if a0 < -1 or a0 > 1:  # Checking a0 is in correct range
            print("a0 out of range {-1, 1}")
            return 0
        # Checking object has spin_field attribute
        try:
            self.asym = float(a0 * np.cos(self.spin_field_angle * self.lifetime))
        except AttributeError as e:
            print(f"{e}: must be defined before calling set_asym")
            raise

    def set_spin_field_angle(self, field_dir):
        """Sets angle between external field and spin direction in radians"""
        self.spin_field_angle = func.get_angle(self.spin_dir, field_dir)

    def set_kubo_toyabe(self, width):
        """Sets kubo-toyabe from equation"""
        self.kt = (1 / 3) + ((2 / 3) * (1 - ((width ** 2) * (self.lifetime ** 2))) *
                             np.exp(-0.5 * (width ** 2) * (self.lifetime ** 2)))

    def feel_dipole(self, dipole):

        try:
            self.field += dipole.get_mag_field(self.location)
        except AttributeError:
            self.field = dipole.get_mag_field(self.location)
        field_mag = func.get_mag(self.field)
        field_dir = func.get_unit_vector(self.field)
        self.apply_field(field_dir, field_mag)
        # print(f"Muon now feels {dipole}")

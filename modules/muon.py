import numpy as np
import modules.functions as func


class Muon:
    """
    Makes an object representing a muon decaying at a set location in a magnetic field
    """
    MASS = 1.883531627e-28
    MASS_ENERGY = 105.6583745e6
    HALFLIFE = 2.2969811e-6
    GYRO_RATIO = 2 * np.pi * 135.5e6
    DECAY_CONST = np.log(2) / HALFLIFE
    TIME_RESOLUTION = 500
    TIME_SCALE = np.linspace(1e-9, 10e-6, TIME_RESOLUTION)

    def __init__(self, **kwargs):
        """
        Initialises each muon object with a lifetime and optional kwargs

        :param dict kwargs: Additional properties and values to give to muon
        """
        self.set_lifetime(np.random.rand(1))
        self.spin_dir = np.array([0, 0, -1])
        self.phase = 0
        # kwargs last so all previous properties can be overwritten if needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def __str__(self):
        """
        :rtpe: str
        :return: list of all attributes and values for self
        """
        return "\n".join([": ".join((str(att), str(getattr(self, att)))) for att in vars(self)])

    @property
    def __repr__(self):
        return f"Muon at {self.location}"

    def apply_field(self, field_dir=[1, 0, 0], field_strength=1e-3, random_phase=False):
        """
        Sets muon attributes if field is present

        :param array field_dir: (x,y,z) unit vector for field
        :param float field_strength: Strength of field at muon
        :param bool random_phase: Choose to randomise muon phase
        :rtype: setter
        :return: None
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
        self.lifetime = -(np.log(U)) / Muon.DECAY_CONST

    def randomise_phase(self):
        """Sets phase to random radian"""
        self.phase = np.random.uniform(0, 2 * np.pi)

    def set_larmor(self, field_mag):
        """
        Sets larmor to radians per second
        :param float field_mag: Magnitude of field
        """
        self.larmor = abs(field_mag) * Muon.GYRO_RATIO

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

    def set_spin_field_angle(self, field):
        """
        Sets angle between external field and spin direction in radians
        :param array field: Magnetic field vector
        :rtype: setter
        """
        self.spin_field_angle = func.get_angle(self.spin_dir, field)

    def set_kubo_toyabe(self, width):
        """Sets kubo-toyabe from equation"""
        self.kt = (1 / 3) + ((2 / 3) * (1 - ((width ** 2) * (self.lifetime ** 2))) *
                             np.exp(-0.5 * (width ** 2) * (self.lifetime ** 2)))

    def feel_dipole(self, dipole):
        """
        Adds field contribution from dipole
        :param object dipole: Dipole for muon to respond to
        :rtype: setter
        """
        try:
            self.field += dipole.get_mag_field(self.location)
        except AttributeError:
            # Catch error if this is first field muon "feels"
            self.field = dipole.get_mag_field(self.location)

    def full_relaxation(self, field):
        """
        Returns the polarisation of the muon against time

        :rtype: array
        :return: Polarisation over TIME_SCALE
        """
        self.set_spin_field_angle(field)
        theta = self.spin_field_angle
        H = func.get_mag(field)
        t = Muon.TIME_SCALE
        y = Muon.GYRO_RATIO
        polarisation = np.cos(theta) ** 2 + (np.sin(theta) ** 2) * np.cos(y * H * t)
        # Set polarisation to zero if muon has decayed
        polarisation = np.where(Muon.TIME_SCALE < self.lifetime, polarisation, np.nan)
        return polarisation

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    np.random.seed(1)
    N = 3000
    particles = [Muon() for _ in range(N)]
    particles[0].lifetime = Muon.TIME_SCALE[-1]
    field_x = np.random.normal(8e-3, 1e-3, N)
    field_y = np.zeros_like(field_x)
    field_z = np.zeros_like(field_x)
    fields = list(zip(field_x, field_y, field_z))

    # Get each muons polarisation
    relaxations = np.array([p.full_relaxation(fields[i]) for i, p in enumerate(particles)])

    # Normalise sum
    overall = np.nansum(relaxations, axis=0) / N

    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(Muon.TIME_SCALE, relaxations[i], alpha=0.5, lw=0.5)
    ax.plot(Muon.TIME_SCALE, overall, lw=2, c="k")
    ax.set_xlim(Muon.TIME_SCALE[0], Muon.TIME_SCALE[-1])
    # Add legend
    legend_handles = {Line2D([0], [0],
                             color="g", markerfacecolor="w",
                             label="Individual muons"),
                      Line2D([0], [0],
                             color="k", markerfacecolor="k",
                             label="Summed relaxation functions")}
    ax.legend(handles=legend_handles, loc="upper right")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))
    ax.s
    plt.show()

    # polars = np.zeros_like(particles, dtype=float)
    # lifes = np.zeros_like(polars)
    #
    # for i, p in enumerate(particles):
    #     p.apply_field(fields[i])
    #     polars[i] = p.polarisation
    #     lifes[i] = p.lifetime
    #
    # fig, (ax, ax2) = plt.subplots(2, 1)
    # ax.scatter(lifes, polars)
    # ax.set_xlim(func.get_limits(lifes))
    # ax2.hist(lifes, bins=100)
    # plt.show()

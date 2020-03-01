import time
import numpy as np
import modules.functions as func
import modules.grid as grid

class Dipole():
    """
    Creates a dipole object to be placed on a Modules.Grid() point
    """
    count = 0

    def __init__(self, orientation=0, coord=[0, 0, 0], strength=1e-3, **kwargs):
        """
        :param float orientation: Angle of dipole in degrees (+ve x = 0)
        :param array location: Location of dipole [x, y, z]
        :param float strength: Magnetic field strength (Tesle)
        :param dict kwargs: attributes and their values to set
        """
        self.coord = np.array(coord)
        self.orientation_d = orientation
        self.orientation_r = np.deg2rad(orientation)
        self.strength = strength
        self.moment = np.array([strength * np.cos(self.orientation_r),
                                strength * np.sin(self.orientation_r)])
        # Set attributes from **kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if len(coord) == 3:
            self.moment = np.append(self.moment, 0)
        Dipole.count += 1

    def __repr__(self):
        """Sets string representation of the instance"""
        return ("Dipole object:  coord: {}\n\
                Orientation: {:.2f} degrees\n\
                Strength: {:.3e} T\n\
                Moment: {}".format(self.coord, self.orientation_d,
                                   self.strength, self.moment))

    def get_mag_field(self, target):
        """
        :param array target: Target location [x, y, z]
        :return: array of field values at target location
        """
        # Check that coordinates are same dimension
        if not len(target) == len(self.location):
            raise ValueError("Dimensions of target and self.location to not match")
        temp_moment = self.moment[:len(target)]  # Ensure moment is correct dimension
        mag_perm = 10e-7  # Cancel constant terms to get mag_perm as only constant
        relative_loc = np.subtract(np.array(target), self.location)
        magnitude = func.get_mag(relative_loc)

        return mag_perm * (
                   (3 * relative_loc * (np.dot(temp_moment, relative_loc))
                    / (magnitude ** 5))
                    - (temp_moment / (magnitude ** 3))
                    )

    def get_relative_loc(self, other):
        """
        :param object other: Location of "other" dipole location
        :return vector: Relative vector
        """
        return other.location - self.location


def set_dir_2d(vector):
    """
    :param array vector: (x, y)
    :rtype: float
    :return: angle of 2d vector
    """
    return np.arctan(vector[1] / vector[0])
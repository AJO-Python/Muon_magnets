import time
import numpy as np
import modules.functions as func

class Dipole():
    """
    Creates a dipole with a defined coordinate on a grid and a magnetic moment
    The magnetic field can be calculated at any point around dipole with
    get_mag_field()
    The magnetic moment will always point along the along the orientation direction
    with a magnitude = strength
    """
    count = 0

    def __init__(self,
                 orientation=0,
                 strength=1e-3,
                 **kwargs):
        """
        :param float orientation: Angle of dipole in degrees (+ve x = 0)
        :param array coord: Index of dipole [x, y, z]
        :param float strength: Magnetic field strength (Tesla)
        :param dict kwargs: attributes and their values to set
        """
        self.orientation_d = orientation
        self.orientation_r = np.deg2rad(orientation)
        self.strength = strength
        self.moment = np.array([strength * np.cos(self.orientation_r),
                                strength * np.sin(self.orientation_r), 0])
        # Set attributes from **kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        Dipole.count += 1

    def __repr__(self):
        """Sets string representation of the instance"""
        return (f"""Dipole object:\n
                location: {self.location}\n\
                Orientation: {self.orientation_d:.2f} degrees\n\
                Strength: {self.strength:.3e} T\n\
                Moment: {self.moment}""")

    def get_mag_field(self,  target):
        """
        :param array target: Target location [x, y, z]
        :return: array of field values at target location
        """
        # Check that coordinates are same dimension
        if not len(target) == len(self.location):
            raise ValueError("Dimensions of target and self.location to not match")
        temp_moment = self.moment[:len(target)]  # Ensure moment is correct dimension
        mag_perm = 1.257e-6  # Cancel constant terms to get mag_perm as only constant
        relative_loc = np.subtract(np.array(target), self.location)
        magnitude = func.get_mag(relative_loc)

        field = np.array(mag_perm * (
                (3 * relative_loc * (np.dot(temp_moment, relative_loc))
                 / (magnitude ** 5))
                - (temp_moment / (magnitude ** 3))
        ))
        return field

    def get_relative_loc(self, other):
        """
        :param object other: Location of "other" dipole location
        :return vector: Relative vector
        """
        return other.location - self.location

    def fancy_mag_field(self, r):
        r0 = np.array(self.location)
        m = self.moment

        R = np.subtract(np.transpose(r), r0).T
        # assume that the spatial components of r are the outermost axis
        norm_R = np.sqrt(np.einsum("i...,i...", R, R))
        # calculate the dot product only for the outermost axis,
        # that is the spatial components
        m_dot_R = np.tensordot(m, R, axes=1)

        # tensordot with axes=0 does a general outer product - we want no sum
        B = (3 * m_dot_R * R / (norm_R ** 5)) - (np.tensordot(m, 1 / (norm_R ** 3), axes=0))

        # include the physical constant
        B *= 1e-7
        return B[0], B[1], B[2]


def set_dir_2d(vector):
    """
    :param array vector: (x, y)
    :rtype: float
    :return: angle of 2d vector
    """
    return np.arctan(vector[1] / vector[0])

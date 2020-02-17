import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from Modules.muon import Muon
from Modules.positron import Positron
import Modules.functions as func

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Setting graph defaults to avoid repetition on each plot
mpl.rcParams["axes.formatter.limits"] = -2, 2  # Sets xticks to use exponents
mpl.rcParams["axes.grid"] = True  # Turns grid on
mpl.rcParams["legend.loc"] = "best"  # Turns legend on and autoplaces i

class Dipole(object):
    """
    Creates dipole:
    location = (pos_x, pos_y)
    orientation and length = pole_separation
    """

    count = 0
    
    def __init__(self, orientation, location, strength):
        """
        orientation: degrees (float)
        strength: tesla (float)
        location: list
        """
        self.location = np.array(location)
        self.orientation_d = orientation
        self.orientation_r = np.deg2rad(orientation)
        self.strength = strength
        self.moment = np.array([strength * np.cos(self.orientation_r),
                                strength * np.sin(self.orientation_r), 0])
        Dipole.count += 1

    def __repr__(self):
        """Sets string representation of the instance"""
        return ("Dipole object:  Location: {}\n\
                Orientation: {:.2f} degrees\n\
                Strength: {:.3e} T\n\
                Moment: {}".format(self.location, self.orientation_d,
                                    self.strength, self.moment))

    def get_mag_field(self, target):
        """Gets magnetic field at target location (x, y, z)"""
        # Check that coordinates are same dimension
        if not len(target)==len(self.location):
            raise ValueError("Dimensions of target and self.location to not match")
        mag_perm = 10**-7  # Cancel constant terms to get mag_perm as only constant
        relative_loc = np.subtract(np.array(target), self.location)
        magnitude = func.get_mag(relative_loc)
        return mag_perm * (
                (3*relative_loc*(np.dot(self.moment, relative_loc)) / (magnitude**5))
                - (self.moment / (magnitude**3))
                )

    def get_relative_loc(self, other):
        return other.location - self.location


class Grid(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.add_points(width, height)

    def __repr__(self):
        # widthheight = f"Width: {self.width}, Height: {self.height}"
        # rowscols = f"x points: {self.cols}, y points: {self.rows}"
        # return widthheight + "\n" + rowscols
        return str(self.points)

    def add_to_point(self, x, y, new_value):
        # Makes addition to preexisting cell
        cur_value = self.get_point(x, y)
        self.change_point([x, y], cur_value + new_value)

    def add_points(self, x_points, y_points):
        self.points = np.zeros([x_points, y_points], dtype=object)

    def convert_y(self, y):
        return int(self.height - 1 - y)

    def change_point(self, location, value):
        x, y = location
        y = self.convert_y(y)
        self.points[y][x] = value

    def all_coords(self):
        for x in range(self.width):
            for y in range(self.height):
                yield (x, y)

    def get_point(self, x, y):
        y = self.convert_y(y)
        return self.points[y][x]


def set_dir_2d(vector):
    return np.arctan(vector[1]/vector[0])


def set_field_values(x_array, y_array, dipole_grid, resolution=10):
    # Loop over points in grid and get field at each point
    field_x = x_array[::resolution]
    field_y = y_array[::resolution]
    field = Grid(len(field_x), len(field_y))
    for i, x in enumerate(field_x):
        for j, y in enumerate(field_y):
            for coord in dipole_grid.all_coords():
                target_dipole = dipole_grid.get_point(*coord)
                field_to_add = target_dipole.get_mag_field(target=[x, y, 100e-6])
                field.add_to_point(i, j, field_to_add)
    return field


def create_dipole_grid(x_array, y_array, strength=1e-3, dipole_spacing=40, buffer=20, random_angle=False, angle=0, field_res=1):
    """
    Creates a grid of dipoles
    Places a dipole every {dipole_spacing} on the x/y array with a buffer around the edges
    """
    start = time.time()
    buffer = max(1, buffer)  # Ensure no zero or negative buffer values
    dipole_x_pos = x_array[buffer:-buffer:dipole_spacing]
    dipole_y_pos = y_array[buffer:-buffer:dipole_spacing]

    dipole_grid = Grid(len(dipole_x_pos), len(dipole_y_pos))
    for i, x in enumerate(dipole_x_pos):
        for j, y in enumerate(dipole_y_pos):
            if random_angle:
                angle = np.random.randint(0, 361)
            print(i, j)
            dipole_grid.change_point((i, j), Dipole(orientation=angle,location=[x, y, 0],strength=strength))
    mid = time.time()
    print("Made {} dipoles in {:.3}s".format(Dipole.count, mid-start))
    return dipole_grid


if __name__ == "__main__":
    Nx, Ny = 300, 300
    side_length = 2e-3
    x_array = np.linspace(-side_length, side_length, Nx)
    y_array = np.linspace(-side_length, side_length, Ny)

    dipoles_aligned = create_dipole_grid(x_array, y_array,
                                         dipole_spacing=10,
                                         random_angle=False,
                                         strength=1e-2,
                                         buffer=0)
    dipoles_random = create_dipole_grid(x_array, y_array,
                                        dipole_spacing=10,
                                        random_angle=True,
                                        strength=1e-2,
                                        buffer=0)

    print(dipoles_random)
"""
    plt.figure
    plt.streamplot(x_array, y_array, field_x, field_y, density=1)
    plt.scatter(0, 0, s=100, c="r", alpha=0.5, marker="o")
    #plt.plot([0, 0], [0, dip_1.length])
    plt.xlim(-side_length, side_length)
    plt.ylim(-side_length, side_length)
    plt.show()
"""
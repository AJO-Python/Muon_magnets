import time
import numpy as np
import Modules.functions as func
import Modules.grid as grid
class Dipole(object):
    count = 0
    def __init__(self, orientation, location, strength):
        """
        orientation: degrees (float)
        strength: Tesla (float)
        location: position vector [x, y, z]
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
        if not len(target) == len(self.location):
            raise ValueError("Dimensions of target and self.location to not match")
        temp_moment = self.moment[:len(target)]
        mag_perm = 10e-7  # Cancel constant terms to get mag_perm as only constant
        relative_loc = np.subtract(np.array(target), self.location)
        magnitude = func.get_mag(relative_loc)
        return mag_perm * (
                (3 * relative_loc * (np.dot(temp_moment, relative_loc)) / (magnitude ** 5))
                - (temp_moment / (magnitude ** 3))
        )

    def get_relative_loc(self, other):
        return other.location - self.location


def set_dir_2d(vector):
    return np.arctan(vector[1]/vector[0])


def set_field_values(x_array, y_array, dipole_grid, resolution=10):
    # Loop over points in grid and get field at each point
    field_x = x_array[::resolution]
    field_y = y_array[::resolution]
    field = grid.Grid(len(field_x), len(field_y))
    for i, x in enumerate(field_x):
        for j, y in enumerate(field_y):
            for coord in dipole_grid.all_coords():
                target_dipole = dipole_grid.get_point(coord)
                field_to_add = target_dipole.get_mag_field(target=[x, y, 100e-6])
                field.add_to_point((i, j), field_to_add)

    return field


def create_dipole_grid(x_array, y_array, strength=1e-3,
                       dipole_spacing=40, buffer=20,
                       random_angle=False, angle=0, field_res=1):
    """
    Creates a grid of dipoles
    Places a dipole every {dipole_spacing} on the x/y array with a buffer
    around the edges
    """
    start = time.time()
    buffer = max(1, buffer)  # Ensure no zero or negative buffer values
    dipole_x_pos = x_array[buffer:-buffer:dipole_spacing]
    dipole_y_pos = y_array[buffer:-buffer:dipole_spacing]

    dipole_grid = grid.Grid(len(dipole_x_pos), len(dipole_y_pos))
    for i, x in enumerate(dipole_x_pos):
        for j, y in enumerate(dipole_y_pos):
            if random_angle:
                angle = np.random.randint(0, 361)
            # print(i, j)
            dipole_grid.set_point((i, j), Dipole(orientation=angle, location=[x, y, 0], strength=strength))
    mid = time.time()
    print("Made {} dipoles in {:.3}s".format(Dipole.count, mid - start))
    Dipole.count = 0
    return dipole_grid

# if __name__ == "__main__":
#     Nx, Ny = 10, 10
#     side_length = 2e-3
#     x_array = np.linspace(-side_length, side_length, Nx)
#     y_array = np.linspace(-side_length, side_length, Ny)
#
#     dipoles_aligned = create_dipole_grid(x_array, y_array,
#                                          dipole_spacing=5,
#                                          random_angle=False,
#                                          strength=1e-2,
#                                          buffer=1)
#     dipoles_random = create_dipole_grid(x_array, y_array,
#                                         dipole_spacing=5,
#                                         random_angle=True,
#                                         strength=1e-2,
#                                         buffer=1)
#     field = set_field_values(x_array, y_array, dipoles_aligned, resolution=2)
#     #field = [dipoles_aligned.get_point(*coord) for coord in dipoles_aligned.all_coords()]
#     #print(field.get_axis_values("all"))
"""
    plt.figure
    plt.streamplot(x_array, y_array, field_x, field_y, density=1)
    plt.scatter(0, 0, s=100, c="r", alpha=0.5, marker="o")
    #plt.plot([0, 0], [0, dip_1.length])
    plt.xlim(-side_length, side_length)
    plt.ylim(-side_length, side_length)
    plt.show()
"""

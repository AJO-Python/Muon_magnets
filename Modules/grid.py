import numpy as np
import Modules.dipole as dip


class Grid():
    """
    Creates a width x height grid that can store objects at all points.
    """

    def __init__(self, width, height):
        """
        :param int width: Number of columns
        :param int height: Number of rows
        :rtype: object
        """
        self.width = int(width)
        self.height = int(height)
        self.add_points(width, height)
        self.size = (self.width, self.height)

    def __repr__(self):
        ret_str = ""
        comma = False
        for x in range(self.width, -1, -1):
            for y in range(0, self.height + 1, 1):
                # Exludes comma for first entry in newline
                if not comma:
                    ret_str = "".join([ret_str, f"({x},{y})"])
                    comma = True
                else:
                    ret_str = ", ".join([ret_str, f"({x},{y})"])
            ret_str += "\n"
            comma = False
        return ret_str

    def add_points(self, x_points, y_points):
        """
        :param int x_points: Number of x points
        :param int y_points: Number of y points
        :return: sets self.points to 2-d array of zeros in shape (x_points, y_points)
        """
        self.points = np.zeros([x_points, y_points], dtype=object)

    def set_point(self, coord, value):
        """
        :rtype: object
        :param tuple coord: (x, y)
        :param object value: Value to store at point
        :return: sets self.points at coord to value
        """
        x, y = coord
        y = self.convert_y(y)
        self.points[y][x] = value

    def add_to_point(self, coord, new_value):
        """
        :param tuple coord: (x, y)
        :param object new_value: Value to add to current point
        :return: Sets self.points += new_value
        """
        # Makes addition to preexisting cell
        cur_value = self.get_point(coord)
        self.set_point(coord, cur_value + new_value)

    def convert_y(self, y):
        """
        :param int y: y coordinate
        :return: Converts from top left indexing to bottom left indexing
        """
        return int(self.height - 1 - y)

    def get_point(self, coord):
        """
        :param tuple coord: (x, y)
        :rtype: object
        :return: Gets the object stored at grid point (x, y)
        """
        x, y = coord
        return self.points[self.convert_y(y)][x]

    def all_coords(self):
        """
        :return: Generator for all (x, y) coords in Grid object
        """
        for x in range(self.width):
            for y in range(self.height):
                yield (x, y)

    def all_values(self):
        """
        :return: Generator for all values stored in Grid
        """
        for coord in self.all_coords():
            yield self.get_point(coord)

    def fill_with_dipoles(self, spacing=3e-6, angle=0,
                          random_angle=False):
        """
        :param float spacing: Centre to centre distance of dipoles
        :param float angle: Orientation of dipole in degrees
        :param bool random_angle: If True provides random angle for each dipole
        :param bool full: If True returns last Dipole object calculated
        :return: Fills field with dipoles
        """
        self.real_size = [val * spacing for val in self.size]
        for coord in self.all_coords():
            pos = spacing
            if random_angle:
                angle = np.random.uniform(0, 360)
            self.set_point(coord,
                           dip.Dipole(orientation=angle,
                                      location=[coord[0] * pos, coord[1] * pos],
                                      strength=1e-4))

    def fill_field_values(self, x_locs, y_locs):
        """
        'Self' must be grid object filled with Dipole objects to use this method
        :param array x_locs: Array of x-locations
        :param array y_locs: Array of y-locations
        :rtype: [array, array]
        :return: Returns x and y field values for all points
        """
        x_len = len(x_locs)
        y_len = len(y_locs)
        Ex = np.zeros([x_len, y_len])
        Ey = np.zeros([x_len, y_len])
        for i, x in enumerate(x_locs):
            print(f"Calculating row... {i}/{x_len - 1}")
            for j, y in enumerate(y_locs):
                for dipole in self.all_values():
                    ex, ey = dipole.get_mag_field([x, y])
                    Ex[j][i] += ex
                    Ey[j][i] += ey
        return Ex, Ey

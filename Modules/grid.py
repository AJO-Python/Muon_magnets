import numpy as np


class Grid(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.add_points(width, height)

    @property
    def __repr__(self):
        # widthheight = f"Width: {self.width}, Height: {self.height}"
        # rowscols = f"x points: {self.cols}, y points: {self.rows}"
        # return widthheight + "\n" + rowscols
        ret_str = ""
        comma = False
        for x in range(self.width, -1, -1):
            for y in range(0, self.height + 1, 1):
                if not comma:
                    ret_str = "".join([ret_str, f"({x},{y})"])
                    comma = True
                else:
                    ret_str = ", ".join([ret_str, f"({x},{y})"])
            ret_str += "\n"
            comma = False
        return ret_str

    def add_points(self, x_points, y_points):
        self.points = np.zeros([x_points, y_points], dtype=object)

    def set_point(self, coord, value):
        x, y = coord
        y = self.convert_y(y)
        self.points[y][x] = value

    def add_to_point(self, coord, new_value):
        # Makes addition to preexisting cell
        cur_value = self.get_point(coord)
        self.set_point(coord, cur_value + new_value)

    def convert_y(self, y):
        return int(self.height - 1 - y)

    def get_point(self, coord):
        """
        :param tuple coord: (x, y)
        :return: object
        Gets the object at grid point (x, y)
        """
        x, y = coord
        return self.points[self.convert_y(y)][x]

    def all_coords(self):
        for x in range(self.width):
            for y in range(self.height):
                yield (x, y)

    def get_axis_values(self, axis):
        if axis.lower() == "all":
            xs, ys, zs = [], [], []
            for coord in self.all_coords():
                print(field)

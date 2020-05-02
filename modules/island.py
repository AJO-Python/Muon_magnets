import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Arrow
from modules.dipole import Dipole
import modules.functions as func

# noinspection PyAttributeOutsideInit

class Island(Dipole):
    counter = 0

    def __init__(self,
                 orientation=0,
                 strength=1e-8,
                 size=(1e-8, 2e-8),
                 location=[0, 0, 0]):
        """
        Creates a 2d dipole pointing along the x-axis

        :param float orientation: Angle of dipole in degrees (+ve x = 0)
        :param array coord: Index of dipole [x, y, z]
        :param float strength: Magnetic field strength (Tesla)
        :param tuple (float, float) size: Y and X length of dipole
        :param tuple (float, float) location: Location of island
        :param dict kwargs: attributes and their values to set
        """
        super().__init__(orientation, strength)
        self.size = np.asarray(size)
        self.area = self.size[0] * self.size[1]
        self.location = location
        self.set_corners()
        self.set_current()
        Island.counter += 1
        self.ID = Island.counter

    def set_corners(self):
        """
        Calculates corners relative to centre
        Sets self.corners and self.path (for use with drawing the islands)

        :return: set self.corners, set self.path
        """
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2

        center_x = self.location[0]
        center_y = self.location[1]
        corners = np.array([
            [center_x - half_width, center_y + half_height],  # top left
            [center_x + half_width, center_y + half_height],  # top right
            [center_x + half_width, center_y - half_height],  # bottom right
            [center_x - half_width, center_y - half_height]  # bottom left
        ])
        self.corners = corners
        self.rotate()
        self.path = mpltPath.Path(self.corners)

    def rotate(self):
        """
        Applies rotation matrix to corners around centre of island
        :return: rotates self.corners
        """
        c, s = np.cos(self.orientation_r), np.sin(self.orientation_r)
        rot_matrix = np.array([[c, -s], [s, c]])
        origin = np.atleast_2d(self.location[:2])
        points = np.atleast_2d(self.corners)
        self.corners = np.squeeze((rot_matrix @ (points.T - origin.T) + origin.T).T)

    def contains_points(self, points):
        inside_indices = self.path.contains_points(points)
        self.inside_muons = sum(inside_indices)
        return inside_indices

    def set_current(self):
        self.current = np.array([self.area * self.moment])

    def get_moment_arrow(self):
        arrow = Arrow(x=self.location[0], y=self.location[1],
                      dx=self.size[0] * np.cos(self.orientation_r) / 2,
                      dy=self.size[1] * np.sin(self.orientation_r) / 2,
                      width=func.get_mag(self.size) / 2,
                      edgecolor=None
                      )
        return arrow

    def get_outline(self):
        """
        Creates a Rectangle patch to outline the Island
        :param float line_width: Width of outline
        :rtype: object
        :return: Rectangle patch
        """
        if not hasattr(self, "line_width"):
            self.line_width = 1

        rectangle = Rectangle(xy=(0, 0),
                              width=self.size[0],
                              height=self.size[1],
                              angle=self.orientation_d,
                              fill=True,
                              alpha=0.4,
                              edgecolor="k",
                              lw=self.line_width)
        rectangle.set_xy(self.corners[3])
        return rectangle


if __name__ == "__main__":
    pass

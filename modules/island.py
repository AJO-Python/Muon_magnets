import numpy as np
from modules.dipole import Dipole
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt


class Island(Dipole):

    def __init__(self, orientation=0, coord=[0, 0, 0],
                 strength=1e-3,
                 size=(1e-8, 2e-8),
                 location=[0, 0, 0]):
        """
        :param float orientation: Angle of dipole in degrees (+ve x = 0)
        :param array coord: Index of dipole [x, y, z]
        :param float strength: Magnetic field strength (Tesla)
        :param tuple (float, float) size: x and y length of dipole
        :param dict kwargs: attributes and their values to set
        """
        super().__init__(orientation, coord, strength)
        self.size = size
        self.location = location
        self.set_corners()

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
        self.corners = self.rotate(corners)
        self.path = mpltPath.Path(self.corners)


    def rotate(self, corners):
        """
        Applies rotation matrix to corners around centre of island

        :param array(vectors) corners: Corner positions to rotate
        :return: rotated corner vectors
        """
        c, s = np.cos(self.orientation_r), np.sin(self.orientation_r)
        rot_matrix = np.array([[c, -s], [s, c]])
        origin = np.atleast_2d(self.location)
        points = np.atleast_2d(corners)
        return np.squeeze((rot_matrix @ (points.T - origin.T) + origin.T).T)

    def contains_points(self, points):
        return self.path.contains_points(points)


if __name__ == "__main__":
    sample = Island(orientation=13, strength=1,
                    size=(0.5, 2), location=[-0.5, -0.5])
    points = np.random.uniform(-1, 1, (10_000, 2))
    colors = np.full(len(points), "g", dtype=str)
    colors[sample.contains_points(points)] = "r"

    plt.scatter(points[:, 0], points[:, 1], c=colors, marker="o", linewidths=0.1)
    plt.legend(loc="best")
    plt.grid()
    plt.axes().set_aspect("equal", "datalim")
    plt.show()

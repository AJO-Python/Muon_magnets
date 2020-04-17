import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Arrow
from modules.dipole import Dipole
import modules.functions as func


class Island(Dipole):
    counter = 0

    def __init__(self,
                 orientation=0,
                 coord=[0, 0, 0],
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
        super().__init__(orientation, coord, strength)
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
                      # dx=self.moment[0] * 10, dy=self.moment[1] * 10,
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
            self.line_width = 0.1

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

    num_muons = 10_0000
    island_size = (1.6e-6, 700e-9)
    grid_size = 10e-6
    num_islands = 10

    # Setup muons
    grid = (-grid_size, grid_size)
    points = np.random.uniform(*grid, (num_muons, 2))
    colors = np.full(len(points), "g", dtype=str)

    # Setup islands
    locations = np.random.uniform(-grid_size / 1.2, grid_size / 1.2, size=(num_islands, 2))
    islands = np.zeros(num_islands, dtype=object)
    for i in range(num_islands):
        temp_island = Island(orientation=np.random.randint(0, 361), strength=8e-8,
                             size=island_size, location=locations[i])
        colors[temp_island.contains_points(points)] = "r"
        islands[i] = temp_island


    def show_on_plot(fig, ax, islands):

        for isle in islands:
            ax.add_patch(isle.get_moment_arrow())
            ax.add_patch(isle.get_outline())
        return fig, ax


    # PLOTTING
    fig, ax = plt.subplots()
    fig, ax = show_on_plot(fig, ax, islands)

    # Add muons
    ax.scatter(points[:, 0], points[:, 1],
               c=colors,
               marker="o",
               s=0.2,
               edgecolors="none")
    # Add legend
    legend_handles = {Line2D([0], [0],
                             color="w", markerfacecolor="r",
                             marker="o", label="Inside"),
                      Line2D([0], [0],
                             color="w", markerfacecolor="g",
                             marker="o", label="Outside")}
    ax.legend(handles=legend_handles, loc="best")

    # Set limits
    ax.set_xlim(func.get_limits(grid))
    ax.set_ylim(func.get_limits(grid))

    # Format plot
    ax.grid()
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.show()

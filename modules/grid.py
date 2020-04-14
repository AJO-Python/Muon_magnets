import numpy as np
import matplotlib.pyplot as plt

import modules.dipole as dip
import modules.functions as func
from modules.island import Island
from modules.ensemble import Ensemble


class Grid():
    """
    Creates a width x height grid that can store objects at all points.
    """

    def __init__(self, config_file="dipole_array_config"):
        parameters = func.load_config(config_file)
        for key, value in parameters.items():
            self.__setattr__(key, value)

        # Setup islands
        self.set_size()
        self.set_count()
        self.set_locations()
        self.create_islands()

    def create_islands(self):
        self.islands = np.zeros(self.count, dtype=object)
        if self.random_orientation:
            self.angles = np.random.random(size=self.count) * 360
        else:
            self.angles = np.full(self.count, self.angle)

        for i in range(self.count):
            temp_island = Island(orientation=self.angles[i], strength=self.strength,
                                 size=self.island_size, location=self.locations[i])
            self.islands[i] = temp_island

    def set_locations(self):
        self.locations_x = np.linspace(-self.width / 2, self.width / 2, self.xnum, endpoint=True)
        self.locations_y = np.linspace(-self.height / 2, self.height / 2, self.ynum, endpoint=True)
        self.locations = []
        for x in self.locations_x:
            for y in self.locations_y:
                self.locations.append((x, y))
        self.locations = np.array(self.locations)

    def set_count(self):
        self.xnum = int(self.width / self.xspacing)
        self.ynum = int(self.height / self.yspacing)
        self.count = self.xnum * self.ynum

    def show_on_plot(self, fig=None, ax=None):
        if not fig and not ax:
            fig, ax = plt.subplots()

        for isle in self.islands:
            ax.add_patch(isle.get_moment_arrow())
            ax.add_patch(isle.get_outline())
        ax.set_xlim(func.get_limits(self.locations_x))
        ax.set_ylim(func.get_limits(self.locations_y))
        ax.set_aspect("equal")
        return fig, ax

    def set_size(self):
        self.island_size = (self.xsize, self.ysize)


if __name__ == "__main__":
    num_muons = 1_0000

    print("Making grid...")
    island_grid = Grid()
    print("Finished grid")

    print("Making ensemble...")
    muon_ensemble = Ensemble(num_muons)
    print("Finished ensemble")

    muon_ensemble.apply_quadtree(island_grid)

    fig, ax = plt.subplots()

    print("Plotting grid...")
    fig, ax = island_grid.show_on_plot(fig, ax)
    print("Finished plotting grid")

    print("Plotting ensemble...")
    fig, ax = muon_ensemble.show_on_plot(fig, ax)
    print("Finished plotting ensemble...")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
    plt.show()

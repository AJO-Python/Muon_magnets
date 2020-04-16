import numpy as np
import matplotlib.pyplot as plt
import os

import modules.dipole as dip
import modules.functions as func
from modules.island import Island
from modules.ensemble import Ensemble


class Grid():
    """
    Creates a width x height grid that can store objects at all points.
    """

    def __init__(self, config_file="dipole_array_config",
                 run_name="",
                 load_only=False):
        """

        :param config_file:
        :param str run_name: Set to overwrite previous run data
        :param bool load_only: If True creates empty class.
        Use with Grid().loader()
        """
        self.run_name = run_name
        if load_only:
            self.loader(self.run_name)
            return

        parameters = func.load_config(config_file)
        for key, value in parameters.items():
            self.__setattr__(key, value)

        # Setup islands
        print("Setting up grid parameters...")
        self.set_size()
        self.set_count()
        self.set_locations()
        self.set_angles()
        if not run_name:
            self.set_run_name()

        print("Creating islands...")
        self.create_islands()
        print("Saving islands...")
        self.save_grid()
        print("Finished.")
        print("===================")

    def create_islands(self):
        """
        Creates array and populates it with Islands
        :return:
        """
        self.islands = np.zeros(self.count, dtype=object)
        for i in range(self.count):
            temp_island = Island(orientation=self.angles[i], strength=self.strength,
                                 size=self.island_size, location=self.locations[i])
            self.islands[i] = temp_island

    def set_locations(self, z_dist=0):
        """
        Creates array of 3d locations to assign to Islands
        :param float z_dist: Z location of dipole layer
        """
        self.locations_x = np.linspace(-self.width / 2, self.width / 2,
                                       self.xnum, endpoint=True)
        self.locations_y = np.linspace(-self.height / 2, self.height / 2,
                                       self.ynum, endpoint=True)
        self.locations = []
        for x in self.locations_x:
            for y in self.locations_y:
                self.locations.append((x, y, z_dist))
        self.locations = np.array(self.locations)

    def set_count(self):
        """
        Sets the number of rows (xnum), cols (ynum) and total number of islands (count)
        """
        self.xnum = int(self.width / self.xspacing)
        self.ynum = int(self.height / self.yspacing)
        self.count = self.xnum * self.ynum

    def set_angles(self):
        if self.random_orientation:
            self.angles = np.random.random(size=self.count) * 360
        else:
            self.angles = np.full(self.count, self.angle)

    def set_size(self):
        self.island_size = (self.xsize, self.ysize)

    def set_generic(self, name, value):
        for isle in self.islands:
            setattr(isle, name, value)

    def set_run_name(self):
        made_dir = False
        dir_count = 0
        angle_key = "R" if self.random_orientation else "U"

        run_name = f"{self.xnum}X{self.ynum}_{angle_key}_{dir_count}"

        while not made_dir:
            try:
                os.makedirs(f"data/{run_name}")
                made_dir = True
            except OSError:
                dir_count += 1
                if dir_count <= 10:
                    run_name = run_name[:-1] + str(dir_count)
                else:
                    run_name = run_name[:-2] + str(dir_count)
        self.run_name = run_name

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

    def save_grid(self):
        func.save_object(self.run_name, "grid_obj", self.__dict__)

    def loader(self, run_name):
        params = func.load_object(run_name, "grid_obj")
        self.__dict__.update(params)


if __name__ == "__main__":
    num_muons = 1_000
    #run_name = "15X15_R_0"
    run_name = ""
    print("Making grid...")
    island_grid = Grid(run_name=run_name)
    #island_grid = Grid(run_name=run_name, load_only=True)
    print("Finished grid")

    print("Making ensemble...")
    muon_ensemble = Ensemble(num_muons, run_name=island_grid.run_name)
    print("Finished ensemble")

    fig, ax = plt.subplots()

    print("Plotting grid...")
    fig, ax = island_grid.show_on_plot(fig, ax)
    print("Finished plotting grid")

    print("Plotting ensemble...")
    fig, ax = muon_ensemble.show_on_plot(fig, ax)
    print("Finished plotting ensemble...")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
    plt.show()

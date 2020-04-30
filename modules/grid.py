import numpy as np
import matplotlib.pyplot as plt
import os

import modules.dipole as dip
import modules.functions as func
from modules.island import Island
from modules.ensemble import Ensemble


# noinspection PyAttributeOutsideInit
class Grid:
    """
    Creates a width x height grid that can store objects at all points.
    """

    def __init__(self, config_file="dipole_array_config",
                 run_name="",
                 load_only=False):
        """

        :param str config_file: Name of file to get config parameters from
        :param str run_name: If set will overwrite any preexisting run data
        :param bool load_only: If True creates empty class to load data into.
        Use with Grid().loader()
        """
        self.run_name = run_name
        if load_only:
            self.loader(self.run_name)
            return
        self.config_file = config_file
        parameters = func.load_config(self.config_file)
        for key, value in parameters.items():
            self.__setattr__(key, value)

        # Setup islands
        print("Setting up grid parameters...")
        self.set_size()
        self.set_count()
        self.set_locations()
        self.set_angles()
        print("Creating islands...")
        self.create_islands()
        if not run_name:
            self.set_run_name()
        self.save_config()
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

    def set_locations(self):
        """
        Creates array of 3d locations to assign to Islands
        :param float z_dist: Z location of dipole layer
        """
        self.locations_x = np.linspace(-self.width / 2, self.width / 2,
                                       self.xnum, endpoint=True)
        self.locations_y = np.linspace(-self.height / 2, self.height / 2,
                                       self.ynum, endpoint=True)
        self.locations_z = np.zeros_like(self.locations_x)
        self.locations = np.array([[(x, y, 0)
                                    for x in self.locations_x]
                                   for y in self.locations_y])
        # Flattens array of arrays to get list of all coords
        self.locations = self.locations.reshape(-1, self.locations.shape[-1])
        # self.locations = np.stack([self.locations_x, self.locations_y, self.locations_z], axis=1)

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

    def make_collection(self):
        import matplotlib.collections
        self.patches = np.array([(isle.get_outline(), isle.get_moment_arrow()) for isle in self.islands]).flatten()
        self.collection = matplotlib.collections.PatchCollection(self.patches)

    def show_on_plot(self, fig=None, ax=None):
        if not fig and not ax:
            fig, ax = plt.subplots()
        self.make_collection()
        ax.add_collection(self.collection)
        ax.set_xlim(func.get_limits(self.locations_x))
        ax.set_ylim(func.get_limits(self.locations_y))
        ax.set_aspect("equal")
        return fig, ax

    def save_config(self):
        cwd = os.getcwd()
        save_folder = f"data/{self.run_name}/grid_config.txt"
        config_location = f"config/{self.config_file}.txt"
        save_path = os.path.join(cwd, save_folder)
        copy_path = os.path.join(cwd, config_location)
        os.popen(f"cp {copy_path} {save_path}")

    def save_grid(self):
        func.save_object(self.run_name, "grid_obj", self.__dict__)

    def loader(self, run_name):
        params = func.load_object(run_name, "grid_obj")
        self.__dict__.update(params)


if __name__ == "__main__":
    run_name = ""
    print("Making grid...")
    island_grid = Grid(run_name=run_name)


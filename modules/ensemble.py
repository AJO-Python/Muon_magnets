import numpy as np
import matplotlib.pyplot as plt
import time

import modules.functions as func
from modules.muon import Muon
from modules.multi_process import MP_fields


class Ensemble():

    def __init__(self, N=10_000, spread_values={}, run_name="", load_only=False):
        """

        :param N:
        :param dict gauss_widths: Dictionary of width values
        :param dict means: Dictionary of mean locations
        :param run_name:
        :param load_only:
        """
        self.run_name = run_name

        if load_only:
            self.loader(self.run_name)
            return
        self.N = N
        self.spread_values = spread_values
        self.create_locations()
        self.muons = np.array([Muon(loc=self.loc[i]) for i in range(self.N)])
        self.save_ensemble()

    def create_locations(self):

        loc_x = np.random.normal(loc=self.spread_values["x_mean"],
                                 scale=self.spread_values["x_width"],
                                 size=(self.N))
        loc_y = np.random.normal(loc=self.spread_values["y_mean"],
                                 scale=self.spread_values["y_width"],
                                 size=(self.N))
        loc_z = np.random.normal(loc=self.spread_values["z_mean"],
                                 scale=self.spread_values["z_width"],
                                 size=(self.N))
        self.loc = np.stack([loc_x, loc_y, loc_z], axis=1)

    @property
    def xloc(self):
        return self.loc[:, 0]

    @property
    def yloc(self):
        return self.loc[:, 1]

    @property
    def zloc(self):
        return self.loc[:, 2]

    def filter_in_dipoles(self, grid):
        pass

    def set_generic(self, name, value):
        for particle in self.muons:
            setattr(particle, name, value)

    def save_ensemble(self):
        func.save_object(self.run_name, "ensemble_obj", self.__dict__)

    def loader(self, run_name):
        params = func.load_object(run_name, "ensemble_obj")
        self.__dict__.update(params)

    def set_relaxations(self, fields):
        self.relaxations = np.array([p.full_relaxation(fields[i], decay=False) for i, p in enumerate(self.muons)])

    def show_on_plot(self, fig=None, ax=None, thin=1):
        if not fig and not ax:
            fig, ax = plt.subplots()
        for muon in self.muons[::thin]:
            ax.scatter(muon.loc[0], muon.loc[1], s=1, c="g", alpha=0.5)
        return fig, ax

    def calculate_fields(self, grid, silent=False):

        if silent:
            MP_fields(self.run_name, self.muons, grid.islands)
        else:
            print("Starting multiprocessing...")
            start = time.time()
            MP_fields(self.run_name, self.muons, grid.islands)
            end = time.time()
            print(f"Time taken: {end - start}")

    def load_fields(self):
        self.fields, self.field_dict = func.load_fields(self.run_name)

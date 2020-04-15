import numpy as np
import matplotlib.pyplot as plt

import modules.functions as func
from modules.muon import Muon


class Ensemble():

    def __init__(self, N, guass_width=10e-6, run_name="", load_only=False):

        self.run_name = run_name

        if load_only:
            self.loader(self.run_name)
            return
        self.N = N
        self.width = guass_width
        self.create_locations()
        self.muons = np.array([Muon(loc=self.locations[i]) for i in range(self.N)])
        self.save_ensemble()

    def create_locations(self):
        locations = np.random.normal(loc=0, scale=self.width, size=(self.N, 2))
        self.locations = np.insert(locations, 2, 0, axis=1)

    def filter_in_dipoles(self, grid):
        pass

    def save_ensemble(self):
        func.save_object(self.run_name, "ensemble_obj", self.__dict__)

    def loader(self, run_name):
        params = func.load_object(run_name, "ensemble_obj")
        self.__dict__.update(params)

    def show_on_plot(self, fig=None, ax=None):
        if not fig and not ax:
            fig, ax = plt.subplots()
        for muon in self.muons:
            ax.scatter(muon.loc[0], muon.loc[1], s=1, c="g", alpha=0.5)
        return fig, ax

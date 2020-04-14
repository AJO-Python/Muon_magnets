import numpy as np
import matplotlib.pyplot as plt

import modules.functions as func
from modules.muon import Muon


class Ensemble():

    def __init__(self, N, guass_width=10e-6):
        self.N = N
        self.width = guass_width
        # self.muons = np.zeros(N, dtype=object)
        self.create_locations()
        self.muons = np.array([Muon(loc=self.locations[i]) for i in range(self.N)])

    def create_locations(self):
        locations = np.random.normal(loc=0, scale=self.width, size=(self.N, 2))
        self.locations = np.insert(locations, 2, 0, axis=1)

    def show_on_plot(self, fig=None, ax=None):
        if not fig and not ax:
            fig, ax = plt.subplots()
        for muon in self.muons:
            ax.scatter(muon.loc[0], muon.loc[1], s=1, c="g", alpha=0.5)
        return fig, ax

    def filter_in_dipoles(self, grid):
        pass

    def apply_quadtree(self, grid):

        for muon in self.muons:
            muon.test = True

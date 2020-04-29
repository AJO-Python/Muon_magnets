import numpy as np
import matplotlib.pyplot as plt
import time

import modules.functions as func
from modules.muon import Muon
from modules.multi_process import MP_fields


class Ensemble():

    def __init__(self, N=10_000, loc_spread_values={}, run_name="", load_only=False):
        """

        :param int N: Size of ensemble
        :param dict loc_spread_values: Dictionary of mean and standard deviation
                                        for location values
        :param str run_name: File name to save to
        :param bool load_only: If True loads data from $run_name
        """
        self.run_name = run_name
        if load_only:
            self.loader(self.run_name)
            return

        if not loc_spread_values:
            self.loc_spread_values = dict(x_width=0, y_width=0, z_width=0,
                                          x_mean=0, y_mean=0, z_mean=0)
        else:
            self.loc_spread_values = loc_spread_values
        self.N = N
        self.create_locations()
        self.muons = np.array([Muon(loc=self.loc[i]) for i in range(self.N)])
        self.save_ensemble()

    @property
    def xloc(self):
        return self.loc[:, 0]

    @property
    def yloc(self):
        return self.loc[:, 1]

    @property
    def zloc(self):
        return self.loc[:, 2]

    def create_locations(self):
        try:
            loc_x = np.random.normal(loc=self.loc_spread_values["x_mean"],
                                     scale=self.loc_spread_values["x_width"],
                                     size=(self.N))
            loc_y = np.random.normal(loc=self.loc_spread_values["y_mean"],
                                     scale=self.loc_spread_values["y_width"],
                                     size=(self.N))
            loc_z = np.random.normal(loc=self.loc_spread_values["z_mean"],
                                     scale=self.loc_spread_values["z_width"],
                                     size=(self.N))
            self.loc = np.stack([loc_x, loc_y, loc_z], axis=1)
        except KeyError:
            self.loc = np.array([])

    def set_relaxations(self):
        self.relaxations = np.array(
            [p.full_relaxation(decay=False) for p in self.muons])
        self.overall_relax = np.mean(self.relaxations, axis=0, dtype=np.float64)

    def calculate_fields(self, grid, silent=False):

        if silent:
            MP_fields(self, grid, silent=True)
        else:
            print("Starting multiprocessing...")
            start = time.time()
            MP_fields(self, grid)
            end = time.time()
            print(f"Time taken: {end - start}")

    def add_field(self, add_field):
        """Adds a single field to all muons in ensemble"""
        if hasattr(self, "fields"):
            for i, f in enumerate(self.fields):
                self.fields[i] = np.add(f, add_field)
        else:
            self.fields = np.array([add_field for _ in range(self.N)])

        self.magnitudes = np.array([func.get_mag(f) for f in self.fields])
        self.create_field_dict()

    def create_field_dict(self):
        self.field_dict = {"total": self.magnitudes,
                           "x": self.fields[:, 0],
                           "y": self.fields[:, 1],
                           "z": self.fields[:, 2]}

    def random_fields(self, width=100e-6):
        self.field_width = width
        self.fields = np.random.normal(loc=0, scale=width, size=(self.N, 3))
        self.magnitudes = np.array([func.get_mag(f) for f in self.fields])
        self.create_field_dict()

    def chunk_for_processing(self):
        self.chunks = np.array_split(self.muons, 16)

    def filter_in_dipoles(self, grid):
        pass

    def set_generic(self, name, value):
        for particle in self.muons:
            setattr(particle, name, value)

    def save_ensemble(self):
        func.save_object(self.run_name, "ensemble_obj", self.__dict__)

    def load_fields(self):
        self.fields, self.field_dict = func.load_fields(self.run_name)

    def loader(self, run_name):
        params = func.load_object(run_name, "ensemble_obj")
        self.__dict__.update(params)

    def show_on_plot(self, fig=None, ax=None, thin=1):
        if not fig and not ax:
            fig, ax = plt.subplots()
        # for muon in self.muons[::thin]:
        ax.scatter(self.xloc, self.yloc, s=1, c="g", alpha=0.5)
        return fig, ax

    def plot_relax_fields(self, save=True):
        from scipy.optimize import curve_fit
        from modules.model_equations import static_GKT

        popt, pcov = curve_fit(static_GKT, Muon.TIME_ARRAY,
                               self.overall_relax, p0=1e-4)

        # Setup subplots
        plt.figure(figsize=func.set_fig_size(subplots=(3, 2)))
        ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 3), (0, 2))
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax4 = plt.subplot2grid((2, 3), (1, 2))

        field_axes = (ax1, ax2, ax3, ax4)
        # Plot individual lines if N is small
        if len(self.relaxations) < 100:
            for i in range(self.N):
                ax0.plot(Muon.TIME_ARRAY, self.relaxations[i], alpha=0.5, lw=0.5)

        # Plot overall relaxation
        ax0.plot(Muon.TIME_ARRAY, self.overall_relax, lw=2, c="k", alpha=0.7, label="Simulation")
        ax0.plot(Muon.TIME_ARRAY, static_GKT(Muon.TIME_ARRAY, *popt), c="r", label="Curve fit")

        ax0.legend(loc="upper right")
        ax0.set_xlim(0, Muon.TIME_ARRAY[-1])
        ax0.grid()
        ax0.set_title("Simulated relaxation function")
        ax0.ticklabel_format(style="sci", axis="x", scilimits=(-6, -6))

        ax1.set_title("Magnitudes of overall field")

        for sub_ax, field in zip(field_axes, self.field_dict.keys()):
            sub_ax.hist(self.field_dict[field], bins=100)
            sub_ax.set_title(f"Magnitudes of {field}")
            sub_ax.set_xlabel("Field strength (T)")
            sub_ax.set_ylabel("Frequency")
            sub_ax.grid()
            sub_ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
        # Add legend
        ax0.axhline(0.3)
        plt.tight_layout(pad=1)
        if save:
            plt.savefig(f"data/{self.run_name}/Relax_fields.png")
        print(f"Calculated width: {float(popt):.2e} +- {float(pcov[0]):.2e}")


if __name__ == "__main__":
    test = Ensemble(N=10)
    test.random_fields()
    test.add_field(np.array([0, 0, 1]))

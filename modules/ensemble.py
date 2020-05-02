import numpy as np
import matplotlib.pyplot as plt
import time
import os

import modules.functions as func
from modules.muon import Muon
from modules.multi_process import MP_fields


# noinspection PyAttributeOutsideInit

class Ensemble():

    def __init__(self, N=10_000, config_file="muon_ensemble_config", run_name="", load_only=False):
        """

        :param int N: Size of ensemble
        :param str config_file: File to get input parameters from
        :param str run_name: File name to save to
        :param bool load_only: If True loads data from $run_name
        """
        self.run_name = run_name
        if load_only:
            self.loader(self.run_name)
            return
        self.config_file = config_file
        self.load_config()
        self.save_config()
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

        except KeyError:
            try:
                width = self.loc_spread_values["width"]
                length = self.loc_spread_values["length"]
                height = self.loc_spread_values["height"]
                loc_x = np.random.uniform(-width/2, width/2, size=self.N)
                loc_y = np.random.uniform(-length / 2, length / 2, size=self.N)
                loc_z = np.full(shape=self.N, fill_value=height)
            except KeyError:
                print("Problem creating muon locations in create_locations()")
                raise
        self.loc = np.stack([loc_x, loc_y, loc_z], axis=1)

    def set_relaxations(self, fields=[]):
        if not fields:
            try:
                self.relaxations = np.array(
                    [p.full_relaxation(self.fields[i], decay=False) for i, p in enumerate(self.muons)])
            except AttributeError:
                self.load_fields()
                self.set_relaxations()
        else:
            self.relaxations = np.array(
                [p.full_relaxation(fields[i], decay=False) for i, p in enumerate(self.muons)])

        self.overall_relax = np.mean(self.relaxations, axis=0, dtype=np.float64)

    def calculate_fields(self, grid, silent=False):

        if silent:
            MP_fields(self.run_name, self.muons, grid.islands)
        else:
            print("Starting multiprocessing...")
            start = time.time()
            MP_fields(self.run_name, self.muons, grid.islands)
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

    def filter_in_dipoles(self, grid):
        pass

    def set_generic(self, name, value):
        for particle in self.muons:
            setattr(particle, name, value)

    def save_ensemble(self):
        func.save_object(self.run_name, "ensemble_obj", self.__dict__)

    def save_config(self):
        cwd = os.getcwd()
        save_folder = f"data/{self.run_name}/muon_config.txt"
        config_location = f"config/{self.config_file}.txt"
        save_path = os.path.join(cwd, save_folder)
        copy_path = os.path.join(cwd, config_location)
        os.popen(f"cp {copy_path} {save_path}")

    def load_config(self):
        load_data = np.loadtxt(f"config/{self.config_file}.txt",
                               delimiter="\n",
                               dtype=str)
        self.loc_spread_values = {}
        for item in load_data:
            name, value = item.split("=")
            self.loc_spread_values[name] = float(value)

    def loader(self, run_name):
        params = func.load_object(run_name, "ensemble_obj")
        self.__dict__.update(params)
        self.run_name = run_name  # Ensures run_name does not get overwritten

    def load_fields(self):
        """
        Loads field experiences by each muon from multiprocessing calculation
        :return: None
        """
        fields = func.load_run(self.run_name, files=["muon_fields"])
        fields = np.array(fields["muon_fields"]["muon_fields"])
        magnitudes = np.array([func.get_mag(f) for f in fields])
        self.fields = fields
        self.magnitudes = magnitudes
        self.create_field_dict()

    def show_on_plot(self, fig=None, ax=None, thin=1):
        """
        Scatter plot of muon locations
        :param object fig: matplotlib fig obj
        :param object ax: matplotlib ax obj
        :param thin: plots muon[::thin]
        :return: fig, ax
        """
        if not fig and not ax:
            fig, ax = plt.subplots(figsize=func.set_fig_size(width="muon_paper"))
        ax.scatter(self.xloc[::thin], self.yloc[::thin], s=1, c="g", alpha=0.5)
        return fig, ax

    def plot_relax_fields(self, save=True, **kwargs):
        """
        Plots the overall relaxation for the ensemble and the field
        distribution graphs.
        Will plot individual relaxations if N < 100
        :param bool save: saves figure in run folder if True
        :param bool add_third_line: Adds a hline at y=1/3 if True
        :return: None
        """
        from scipy.optimize import curve_fit
        from modules.model_equations import static_GKT

        # Setup subplots
        plt.figure(figsize=func.set_fig_size(width="muon_paper",
                                             fraction=1,
                                             subplots=(3, 2)))
        ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((3, 2), (1, 0))
        ax2 = plt.subplot2grid((3, 2), (1, 1))
        ax3 = plt.subplot2grid((3, 2), (2, 0))
        ax4 = plt.subplot2grid((3, 2), (2, 1))

        field_axes = (ax1, ax2, ax3, ax4)
        # Plot individual lines if N is small
        if len(self.relaxations) < 100:
            for i in range(self.N):
                ax0.plot(Muon.TIME_ARRAY, self.relaxations[i], alpha=0.5, lw=0.5)

        # Plot overall relaxation
        ax0.plot(Muon.TIME_ARRAY, self.overall_relax, lw=2, c="k", alpha=0.7, label="Simulation")
        if "curve_fit" in kwargs.items():
            popt, pcov = curve_fit(static_GKT, Muon.TIME_ARRAY,
                                   self.overall_relax, p0=1e-4)
            self.calculated_width = (popt, pcov[0])
            print(f"Calculated width: {float(popt):.2e} +- {float(pcov[0]):.2e}")
            ax0.plot(Muon.TIME_ARRAY, static_GKT(Muon.TIME_ARRAY, *popt), c="r", label="Curve fit")
        if "add_third_line" in kwargs.items():
            ax0.axhline(1 / 3, color="k", linestyle=":", alpha=0.5, label="1/3 tail")

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
        plt.tight_layout(pad=1)
        if save:
            plt.savefig(f"data/{self.run_name}/Relax_fields.pdf",
                        bbox_inches="tight",
                        format="pdf")

    def plot_distribution(self, grid, save=True):
        """
        Plots the ensemble overlaid on the island grid
        Plots the angular distribution of island orientations
        :param object grid: Island()
        :param bool save: Saves plot to run folder if True
        :return: None
        """
        fig = plt.figure(figsize=func.set_fig_size(width="muon_paper",
                                                   fraction=1,
                                                   subplots=(3, 2)))
        fig.suptitle(self.run_name)

        angle_ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        grid_ax = plt.subplot2grid((2, 2), (0, 0))
        XZ_dist_ax = plt.subplot2grid((2, 2), (0, 1))
        XZ_dist_ax.set_aspect("equal")
        grid_ax.set_aspect("equal")
        XZ_dist_ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
        grid_ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
        grid.set_generic("line_width", 1)

        # angle_ax
        angle_ax.hist(grid.angles, bins=36)
        angle_ax.set_xlabel("Rotation of island (degrees)")
        angle_ax.set_ylabel("Frequency")
        angle_ax.set_title("Angle distribution of Dipoles")

        # XZ_dist_ax
        XZ_dist_ax.scatter(self.xloc[::5], self.zloc[::5], s=1, alpha=0.5)
        XZ_dist_ax.set_xlim(func.get_limits(self.xloc))
        XZ_dist_ax.set_ylim(func.get_limits(self.zloc))
        XZ_dist_ax.set_xlabel("X locations")
        XZ_dist_ax.set_ylabel("Z locations")

        # grid_ax
        fig, grid_ax = grid.show_on_plot(fig, grid_ax)
        fig, grid_ax = self.show_on_plot(fig, grid_ax, thin=3)
        grid_ax.set_xlabel("X locations")
        grid_ax.set_ylabel("Y locations")

        fig.tight_layout(pad=1)
        if save:
            fig.savefig(f"data/{self.run_name}/visualise_distribution.pdf",
                        bbox_inches="tight",
                        format="pdf")

if __name__ == "__main__":
    test = Ensemble(N=10, run_name="20X20_R_1")

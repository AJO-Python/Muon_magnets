#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import modules.functions as func

from matplotlib.patches import Ellipse


def calc_field_grid(run_name, dipole_array, edge_buffer, nx, ny):
    """

    :param array dipole_array: Array of dipoles in grid
    :param int nx: Number of x points to calculate field at
    :param int ny: Number of y points to calculate field at
    :return: Saves field values and locations to run folder
    """
    # Setup field grid and calculate values
    field_locations = setup_field(max_loc=dipole_array[-1].loc,
                                  edge_buffer=edge_buffer,
                                  nx=nx, ny=ny)
    field_values = fill_field_values_2d(dipole_array, **field_locations)

    # Save data
    func.save_array(run_name, "fields", **field_values)
    func.save_array(run_name, "locations", **field_locations)


def setup_field(max_loc, edge_buffer, nx, ny):
    """
    :param tuple max_loc: (X, Y) of furthest dipole
    :param float edge_buffer: Distance around edge of grid to calculate field for
    :param int nx: Number of field x points
    :param int ny: Number of field y points
    :rtype: dict
    :return: Dictionary containing coordinates for a field over the dipole array
    """
    x_max, y_max, _ = max_loc
    # Determine region to calculate field lines/plot over
    field_region = [[0 - edge_buffer, x_max + edge_buffer], [0 - edge_buffer, y_max + edge_buffer]]
    field_locations = {"x_vals": np.linspace(*field_region[0], nx),
                       "y_vals": np.linspace(*field_region[1], ny)}
    return field_locations


def fill_field_values_2d(dipoles, x_vals=[], y_vals=[]):
    """
    :param array dipoles: array of dipoles
    :param array x_locs: x locations to calculate field at
    :param array y_locs: y locations to calculate field at
    :rtype: dict
    :return: X and Y field values for a meshgrid of x_locs and y_locs
    """
    x_len = len(x_vals)
    y_len = len(y_vals)
    Ex = np.zeros([x_len, y_len])
    Ey = np.zeros([x_len, y_len])
    for i, x in enumerate(x_vals):
        print(f"Calculating row... {i}/{x_len - 1}")
        for j, y in enumerate(y_vals):
            for dipole in dipoles:
                if dipole.loc == (x, y, 1e-6):
                    continue
                ex, ey, _ = dipole.get_mag_field([x, y, 1e-6])
                Ex[j][i] += ex
                Ey[j][i] += ey
    return {"Ex": Ex, "Ey": Ey}


if __name__ == "__main__":

    do_plot = True
    run_name = "5x5_R_1"
    nx, ny = 100, 100
    buffer = 3e-6

    dipole_data = func.load_run(run_name, files=["dipoles"])
    dipole_data = dipole_data["dipoles"]
    calc_field_grid(run_name,
                    dipole_data["dipoles"],
                    edge_buffer=buffer,
                    nx=nx,
                    ny=ny)

    if do_plot:
        dipole_data, field_data, loc_data = func.load_run(run_name)
        dipole_array = dipole_data["dipoles"]

        plot_density = 4

        # PLOTTING
        fig, ax = plt.subplots()

        # Set colourscale and plot streamplot
        color = 2 * np.log(np.hypot(field_data["Ex"], field_data["Ey"]))

        ax.streamplot(loc_data["x_vals"],
                      loc_data["y_vals"],
                      field_data["Ex"],
                      field_data["Ey"],
                      color=color,
                      linewidth=1,
                      cmap=plt.cm.inferno,
                      density=plot_density,
                      arrowstyle='->',
                      arrowsize=1.5)

        # Add ellipses to mark dipoles
        ells = [Ellipse(xy=dipole.loc,
                        width=700e-9, height=1.6e-6,
                        angle=dipole.orientation_d + 90) for dipole in dipole_array]
        for ellipse in ells:
            ax.add_artist(ellipse)

        # Set graph parameters
        ax.set_xlabel('$x (m)$')
        ax.set_ylabel('$y (m)$')
        ax.set_xlim(min(loc_data["x_vals"]), max(loc_data["x_vals"]))
        ax.set_ylim(min(loc_data["y_vals"]), max(loc_data["y_vals"]))
        ax.set_title(f"{run_name} grid")
        ax.set_aspect('equal')
        plt.grid()
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-6, -6))
        plt.savefig(f"images/dipoles/{run_name}.png", bbox_inches="tight")
        plt.show()

# -*-coding: UTF-8-*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import modules.functions as func
from modules.muon import Muon
from modules.grid import Grid
from modules.ensemble import Ensemble
from modules.multi_process import MP_fields
from modules.model_equations import static_GKT

RUN_NAME = "15X15_U_3"

NUM_MUONS = 20000
SPREAD_VALUES = {"x_width": 10e-6, "y_width": 10e-6, "z_width": 10e-6,
                 "x_mean": 0, "y_mean": 0, "z_mean": 100e-6}

island_grid = Grid(run_name=RUN_NAME, load_only=True)
# particles = Ensemble(N=NUM_MUONS, spread_values=SPREAD_VALUES, run_name=RUN_NAME)
particles = Ensemble(run_name=island_grid.run_name, load_only=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(particles.xloc, particles.yloc, particles.zloc, s=0.5)
ax.scatter(island_grid.locations[:, 0],
           island_grid.locations[:, 1],
           island_grid.locations[:, 2], s=1, marker="s")

X = particles.xloc
Y = particles.yloc
Z = particles.zloc
# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

ax.ticklabel_format(style="sci", axis="both", scilimits=(-6, -6))
plt.savefig(f"data/{RUN_NAME}/muon_grid.png", bbox_inches="tight")
plt.show()

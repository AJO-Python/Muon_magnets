# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from modules.functions import load_run

run_name = "5x5_U_1"

data = load_run(run_name, files=["dipoles", "muons"])
dipoles = data["dipoles"]["dipoles"]
muons = data["muons"]["muons"]

muon_xs = [muon.loc[0] for muon in muons if not hasattr(muon, "in_island")]
muon_ys = [muon.loc[1] for muon in muons if not hasattr(muon, "in_island")]

fig = plt.figure()
ax = fig.add_subplot(111)
ells = [Ellipse(xy=dipole.loc,
                width=700e-9, height=1.6e-6,
                angle=dipole.orientation_d + 90) for dipole in dipoles]
for ellipse in ells:
    ax.add_artist(ellipse)

ax.scatter(muon_xs, muon_ys, s=0.1, c="r", alpha=0.6, marker="x")

# Set graph parameters
ax.set_xlabel('$x (m)$')
ax.set_ylabel('$y (m)$')
ax.set_xlim(min(muon_xs)-1e-6, max(muon_xs)+1e-6)
ax.set_ylim(min(muon_ys)-1e-6, max(muon_ys)+1e-6)
ax.set_title(f"{run_name} with normally\ndistributed muon placement")
ax.set_aspect('equal')
plt.grid()
plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
plt.ticklabel_format(axis="y", style="sci", scilimits=(-6, -6))
plt.savefig(f"../images/numerical/{run_name}.png", bbox_inches="tight")
plt.show()
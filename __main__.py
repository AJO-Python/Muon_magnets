import numpy as np
import matplotlib.pyplot as plt

import Modules.functions as func
import Modules.muon as mu
import Modules.dipole as dip
import Modules.grid as grid
import model_equations

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

dipole_1 = dip.Dipole(90, [0, 0], 1)
dipole_2 = dip.Dipole(90, [1, 0], -1)
dipole_3 = dip.Dipole(90, [1, 1], 1)
dipole_4 = dip.Dipole(90, [0, 1], 1)
dipoles = [dipole_1, dipole_2]  # , dipole_3, dipole_4]

# Grid of x, y points
nx, ny = 64, 64
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)
# Electric field vector, E=(Ex, Ey), as separate components
Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
        for d in dipoles:
            ex, ey = d.get_mag_field(target=[x_, y_])
            Ex[j][i] += ex
            Ey[j][i] += ey
            if i == 20 and j == 20:
                print(x_, y_)
                print(ex, ey)

fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the streamlines with an appropriate colormap and arrow style
color = 2 * np.log(np.hypot(Ex, Ey))
ax.streamplot(x, y, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

# Add filled circles for the charges themselves
charge_colors = {True: '#aa0000', False: '#0000aa'}
for d in dipoles:
    ax.add_artist(Circle(d.location, 0.05, color=charge_colors[1 > 0]))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
plt.savefig("Images/Dipoles/double_opp_dipole.png", bbox_inches="tight")
plt.show()

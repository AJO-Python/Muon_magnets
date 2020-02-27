#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:33:25 2019

@author: josh
"""
import numpy as np
import matplotlib.pyplot as plt

from .Modules.muon import Muon
import .Modules.functions as func

# =============================================================================
# Takes the inverse of the positron emission function and displays
# histogram of angular distribution
# Used for generating plots for report - NOT FOR USE IN MAIN CODE
# =============================================================================
def theta_predict(W, a0):
    value = (W-1) / a0
    if value >= 1 or value <= -1:
        print("np.arccos() can only take {-1, 1}")
        print("w:", W)
        print("value", value)
        return None
    """
    x2 because arccos returns {0, pi}
    output should be {0, 2pi}
    """
    return 2*np.arccos((W-1) / a0)

N = 100000
angles = np.zeros(N)

for i in range(N):
    """
    Range {0, 2}
    By symmetry of cyclic function
    """
    U = np.random.uniform(0, 2)
    try:
        angles[i] = theta_predict(U, -1)
    except:
        print(f"i: {i}")
        print(f"U: {{U}}")
        print(f"angle: {angles[i]}")
        raise


fig = plt.figure()
ax = plt.subplot(111, projection="polar")
ax.hist(angles, bins=200, histtype="stepfilled")
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(["0", "$\pi/2$", "$\pi$", "$3\pi/2$", "",])
ax.set_yticks(np.linspace(0, 1000, 5))
ax.set_title("Positron emission frequency as a function of angle (N={:.0e})".format(N))
ax.annotate(s="Taking U over {0, 2}\nTaking 2*arccos()", xy=(0.7, 0.1), xycoords="figure fraction")
plt.savefig("Images/Positron_emmision_whole.png")

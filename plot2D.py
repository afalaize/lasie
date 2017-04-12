# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:25:57 2017

@author: afalaize
"""

import matplotlib.pyplot as plt
import numpy as np
from pypod.grids.tools import buildGrid, grid2mesh

g, h = buildGrid([(0, 1), (0, 2)], [0.01, 0.015])
m = grid2mesh(g)

nx, nc = m.shape

v = np.zeros(nx)

for i, (x1, x2) in enumerate(m):
    v[i] = np.sin(np.pi*x1)*np.sin(np.pi*x2)

plt.imshow(v.reshape(g.shape[1:]))
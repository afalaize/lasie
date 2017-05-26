# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 10:48:10 2017

@author: afalaize
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from ..config import ORDER


def generate(minmax, h=1.):
    """
Return an N-dimensional regular grid.

Parameters
-----------
minmax: iterable
    N tuples of 2 floats: minmax = [(x1_min, x1_max), ..., (xN_min, xN_max)]
    with (xi_min, xi_max) the grid limits over the i-th axis.

h: float or iterable
    Grid spacing(s). If h is a float, the grid spacing is the same over every
    axis. If h is an iterable with length N, h[i] is the grid spacing over
    axis i.

Return
-------
grid: array
    A regular (1+N)-dimensional grid. E.g with N=3, grid[c, i, j, k] is
    the component 'c' of the coordinates of the point at position (i, j, k).

h : tuple of floats
    The N space steps for the N directions.
    """
    x_grid = list()
    h_grid = list()
    for i, xminmax in enumerate(minmax):
        ximin, ximax = xminmax
        hi = h[i] if isinstance(h, (tuple, list)) else h
        nxi = max([1, int((ximax-ximin)/hi)])
        xi_grid = np.linspace(ximin, ximax, nxi)
        if nxi > 1:
            hi_grid = np.diff(xi_grid)[0]
        else:
            hi_grid = 0.
        x_grid.append(xi_grid)
        h_grid.append(hi_grid)
    grid = np.array(np.meshgrid(*x_grid, indexing='ij'))
    return grid, h_grid


def to_mesh(grid):
    """
    Convert a regular (1+N)-dimensional grid to a list of coordinates.

    Usage
    ------
    mesh = to_mesh(grid)

    with mesh.shape = (n_points, n_components)
    and mesh[i, :] the coordinates of point i.

    Info
    -----

    for a grid with shape [3, M, N, P], the mesh is organized as follows:
    mesh = [ [x0, y0, z0],
             [x0, y0, z1],
             [    ...   ],
             [x0, y0, zP],
             [x0, y1, z0],
             [x0, y1, z1],
             [    ...   ],
             [x0, y1, zP],
             [    ...   ],
             [x0, yN, zP],
             [x1, y0, z0],
             [x1, y0, z1],
             [    ...   ],
             [x1, yN, zP],
             [    ...   ],
             [xM, yN, zP] ]
    """
    return grid.reshape((grid.shape[0], np.prod(grid.shape[1:])),
                        order=ORDER).T


def from_mesh(mesh, original_grid_shape):
    return mesh.T.reshape(original_grid_shape, order=ORDER)

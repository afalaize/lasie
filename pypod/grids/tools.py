# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 10:48:10 2017

@author: afalaize
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from ..config import ORDER


def buildGrid(minmax, h=1.):
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

    """
    x_grid = list()
    h_grid = list()
    for i, xminmax in enumerate(minmax):
        ximin, ximax = xminmax
        hi = h[i] if isinstance(h, (tuple, list)) else h
        nxi = int((ximax-ximin)/hi)
        xi_grid = np.linspace(ximin, ximax, nxi)
        hi_grid = np.diff(xi_grid)[0]
        x_grid.append(xi_grid)
        h_grid.append(hi_grid)
    grid = np.array(np.meshgrid(*x_grid, indexing='ij'))
    return grid, h_grid


def grid2mesh(grid):
    """
    Convert a regular (1+N)-dimensional grid to a list of coordinates.

    Usage
    ------
    mesh = grid2mesh(grid)

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


def mesh2grid(mesh, original_grid_shape):
    return mesh.T.reshape(original_grid_shape, order=ORDER)

if __name__ is '__main__':
    grid, h = buildGrid([(0, 1), (0, 1), (0, 1)], [0.5, 0.5, 0.5])
    original_shape = grid.shape
    mesh = grid2mesh(grid)
    new_grid = mesh2grid(mesh, original_shape)
    print(np.sum(np.abs(new_grid - grid)))


    def data_over_mesh():
        data = np.zeros((mesh.shape[0], 1, 3, 1))
        for i, x in enumerate(mesh):
            data[i, 0, :, 0] = [x[0]**2, 2*x[0]**2, 3*x[0]**2]
        return data
    d = data_over_mesh()

    nx, nc = mesh.shape
    out_grad = np.zeros((nx, 1, nc, nc))

    for i in range(3):
        di = d[:, 0, i, 0].reshape(original_shape[1:])
        gi = np.gradient(di)
        for j in range(3):
            out_grad[:, 0, i, j] = gi[j].reshape((np.prod(grid.shape[1:]), ))

    li = [out_grad, ]*5
    final = np.concatenate(li, axis=1)

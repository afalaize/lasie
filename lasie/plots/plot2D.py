# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:25:57 2017

@author: afalaize
"""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
from ..misc.tools import norm

NCOLS = 1
CMAP = 'RdBu_r'
SIZE = None
FORMAT = 'png'
AXES = [0.88, 0.125, 0.01, 0.75]
PARAMS = {'left': 0.05,
          'bottom': 0.05,
          'right': 0.85,
          'top': 0.9,
          'wspace': None,
          'hspace': None}
CBAR = 'global'

OPTIONS = {'ncols': NCOLS,
           'cmap': CMAP,
           'size': SIZE,
           'format': FORMAT,
           'axes': AXES,
           'params': PARAMS,
           'cbar': CBAR,
           }


def plot2d(a, shape, title=None, render='magnitude', savename=None, options=None):
    """
Parameters
----------

a : array_like with shape (nx, ne, nc)
    array to plot where nx is the number of spatial discretization points, ne
    is the number of elements to plot, and nc is the number of spatial 
    components.
    
shape : list of int
    Original grid shape (i.e. as returned by lasis.grids.generate).
    
title : string (optional)

render : 'magnitude' or int (optional)
    Render the magnitude or a specific field component.

savename : string (optional)

options : dictionary (optional)
    options = {'ncols': NCOLS,
               'cmap': CMAP,
               'size': SIZE,
               'format': FORMAT,
               'axes': AXES,
               'params': PARAMS,
               'cbar': CBAR,
               }
    """
    

    opts = OPTIONS
    if options is None:
        options = {}
    opts.update(options)

    nx, nm, nc = a.shape
    nrows = int(np.ceil(nm/float(opts['ncols'])))
    fig = plt.figure(figsize=opts['size'])

    if title is not None:
        plt.suptitle(title)

    all_v = list()
    minmax = (float('Inf'), -float('Inf'))
    for m in range(nm):
        d = a[:, m]
        if render == 'magnitude':
            v = norm(d)
        else:
            assert isinstance(render, int)
            assert 0 <= render < nc
            v = d[:, render]
        minmax = min([minmax[0], min(v)]), max([minmax[1], max(v)])
        all_v.append(v)
    for ind, v in enumerate(all_v):
        v_g = v.reshape(map(int, shape[1:]))
        plt.subplot(nrows, opts['ncols'], ind+1)
        plt.title('${}$'.format(ind))
        im = plt.imshow(v_g.T, cmap=opts['cmap'],
                        vmin=minmax[0], vmax=minmax[1])
        plt.axis('off')
        if opts['cbar'] == 'individual':
            plt.colorbar()
    if not opts['cbar'] == 'individual':
        cbar_ax = fig.add_axes(opts['axes'])
        fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
    plt.tight_layout()
    plt.subplots_adjust(**opts['params'])
    if savename is not None:
        plt.savefig('{}.{}'.format(savename, opts['format']),
                    format=opts['format'])

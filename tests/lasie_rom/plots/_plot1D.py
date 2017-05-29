#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:11:09 2017

@author: Antoine Falaize
"""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
from ..misc.tools import norm

NCOLS = None
SIZE = None
FORMAT = 'png'
AXES = [0.88, 0.125, 0.01, 0.75]
PARAMS = {'left': 0.05,
          'bottom': 0.05,
          'right': 0.85,
          'top': 0.9,
          'wspace': None,
          'hspace': None}

OPTIONS = {'ncols': NCOLS,
           'size': SIZE,
           'format': FORMAT,
           'axes': AXES,
           'params': PARAMS,
           }


def plot1d(A, title=None, savename=None, options=None):
    """
Plot of 1D data array.
    
Parameters
----------

A: array_like with shape (nx, nc, nm)
    array to plot where nx is the number of spatial discretization points, nc 
    is the number of spatial components, and nm is the number of elements to 
    plot. 
    
shape: list of int
    Original grid shape (i.e. as returned by lasis.grids.generate).
    
title: string (optional)

render: 'magnitude' or int (optional)
    Render the magnitude or a specific field component.

savename: string (optional)

options: dictionary (optional)
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

    nx, nc, nm = A.shape
    
    if not nc == 1:
        raise TypeError('For 1D data, the shape along 2nd axis should be 1, got {}'.format(nc))
    
    ncols = opts['ncols']
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(nm)))
    nrows = int(np.ceil(nm/float(ncols)))
    fig = plt.figure(figsize=opts['size'])

    if title is not None:
        plt.suptitle(title)

    all_v = list()
    minmax = (float('Inf'), -float('Inf'))
    for m in range(nm):
        v = A[:, 0, m]
        minmax = min([minmax[0], min(v)]), max([minmax[1], max(v)])
        all_v.append(v)
    for ind, v in enumerate(all_v):
        plt.subplot(nrows, ncols, ind+1)
        plt.plot(v, label=str(ind+1))
        plt.legend()
        plt.ylim(*minmax)
    plt.tight_layout()
    plt.subplots_adjust(**opts['params'])
    if savename is not None:
        plt.savefig('{}.{}'.format(savename, opts['format']),
                    format=opts['format'])

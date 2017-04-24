# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:25:57 2017

@author: afalaize
"""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
from ..misc.tools import norm

NCOLS = 3
CMAP = 'RdBu_r'
SIZE = (6, 8)
FORMAT = 'png'
AXES = [0.88, 0.125, 0.01, 0.75]
PARAMS = {'left': 0.05,
          'bottom': 0.05,
          'right': 0.85,
          'top': 0.9,
          'wspace': None,
          'hspace': None}
CBAR = 'global'


def plot2d(a, shape, title=None, render='module', savename=None):

    nx, nm, nc = a.shape
    nrows = int(np.ceil(nm/float(NCOLS)))
    fig = plt.figure(figsize=SIZE)

    if title is not None:
        plt.suptitle('MODELE COMPLET')

    all_v = list()
    minmax = (float('Inf'), -float('Inf'))
    for m in range(nm):
        d = a[:, m]
        if render == 'module':
            v = norm(d)
        else:
            assert isinstance(render, int)
            v = d[:, render]
        minmax = min([minmax[0], min(v)]), max([minmax[1], max(v)])
        all_v.append(v)
    for ind, v in enumerate(all_v):
        v_g = v.reshape(map(int, shape[1:]))
        plt.subplot(nrows, NCOLS, ind+1)
        plt.title('${}$'.format(ind))
        im = plt.imshow(v_g.T, cmap=CMAP, vmin=minmax[0], vmax=minmax[1])
        plt.axis('off')
        if CBAR == 'individual':
            plt.colorbar()
    if not CBAR == 'individual':
        cbar_ax = fig.add_axes(AXES)
        fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
    plt.tight_layout()
    plt.subplots_adjust(**PARAMS)
    if savename is not None:
        plt.savefig('{}.{}'.format(savename, FORMAT),
                    format=FORMAT)

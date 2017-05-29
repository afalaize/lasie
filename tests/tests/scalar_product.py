#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:06:25 2017

@author: root
"""

import numpy as np
from lasie_rom.operators import scalarproduct

nx = 100

if __name__ == '__main__':
    for nu in ([1, 0], [3, 0], [3, 3]):
        for nv in ([1, 0], [3, 0], [3, 3]):
            nu1, nu2 = nu
            nv1, nv2 = nv
            u = np.random.rand(nx, nu1, nu2) if nu2 != 0 else np.random.rand(nx, nu1)
            v = np.random.rand(nx, nv1, nv2) if nv2 != 0 else np.random.rand(nx, nv1)
            a = scalarproduct(u, v, None)
            print('\n\n'+'*'*50)
            print('nu1 = {}\nnu2 = {}\nnv1 = {}\nnv2 = {}\n'.format(nu1, nu2, nv1, nv2))
            print(a)
            print('Done')


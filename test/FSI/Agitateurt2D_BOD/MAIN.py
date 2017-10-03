#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:58:57 2017

@author: afalaize
"""

from set_parameters import parameters
from run_fenics_simu import fenicsSimulation

import numpy as np

ReMin = 100
ReMax = 500

for Re in np.arange(510, 610, 10):
    parameters['lambda'][3] = Re
    fenicsSimulation()

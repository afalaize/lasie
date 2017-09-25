#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:56:15 2017

@author: afalaize

Main script for running a serie of simulation associated with the 2D Agitator
"""


# Fenics Simulation tools
from FenicsAgitateurt2D import fenicsSimulation

# POD based model order reduction
from PodAgitateur2D import execute_all

# Default parameters
lambda1 = 0.375/1.  # Ellipse principal radius/box side length (d.u.)
lambda2 = .2    # Ellipse shape parameter (d.u.)
lambda3 = 0.     # Ellipse excentricity parameter (d.u.)
lambda4 = 100.    # Reynolds Number (d.u.)


# Loop over a chosen set of parameters
for lambda2 in [0.2, 0.4]:
    for lambda4 in [500., ]:
        parameters, resultsFolderName = fenicsSimulation(lambda1,
                                                         lambda2,
                                                         lambda3,
                                                         lambda4)
        execute_all(parameters, resultsFolderName)
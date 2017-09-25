#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:59:40 2017

@author: afalaize

1) Run the simulation case for the ellipse with imposed velocity

2) Compute the pod absis associated with the snapshots and run the reduce-order
model


"""


# %%
# 1) Run the simulation case for the ellipse with imposed velocity

# Fenics Simulation tools
from fenics_simulation import fenicsSimulation

# Default parameters
lambda1 = 0.375/1.  # Ellipse principal radius/box side length (d.u.)
lambda2 = .2    # Ellipse shape parameter (d.u.)
lambda3 = 0.     # Ellipse excentricity parameter (d.u.)
lambda4 = 500.    # Reynolds Number (d.u.)
eps_tanh = 0.0


parameters, resultsFolderName = fenicsSimulation(lambda1,
                                                 lambda2,
                                                 lambda3,
                                                 lambda4,
                                                 eps_tanh)

# %%
# POD based model order reduction

#from PodAgitateur2D import execute_all
#execute_all(parameters, resultsFolderName)
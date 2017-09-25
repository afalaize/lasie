#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:08:44 2017

@author: afalaize
"""

from FenicsAgitateurt2D import build_parameters, build_resultsFolderName

lambda1 = 0.375/1.  # Ellipse principal radius/box side length (d.u.)
lambda2 = .2    # Ellipse shape parameter (d.u.)
lambda3 = 0.     # Ellipse excentricity parameter (d.u.)
lambda4 = 100.    # Reynolds Number (d.u.)
eps_tanh = 0.0

parameters = build_parameters(lambda1, lambda2, lambda3, lambda4, eps_tanh)
dataname = None
parameters.update({'dataname': dataname})

resultsFolderName = build_resultsFolderName(parameters)

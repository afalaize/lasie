#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:52:34 2017

@author: afalaize
"""

import numpy as np

from lasie_rom.io import hdf

# --------------------------------------------------------------------------- #
# Top level parameters

lambda1 = 0.375/1.  # Ellipse principal radius/box side length (d.u.)
lambda2 = .2    # Ellipse shape parameter (d.u.)
lambda3 = 0.     # Ellipse excentricity parameter (d.u.)
lambda4 = 100.    # Reynolds Number (d.u.)
eps_tanh = 0.0

parameters = {'lambda': (lambda1, lambda2, lambda3, lambda4),  # parameters
              'eps_tanh': eps_tanh
              }

# --------------------------------------------------------------------------- #
# Name for the solution data to retrieve from the vtu files

dataname_vtu = None

if dataname_vtu is None:
    dataname_hdf = None
else:
    dataname_hdf = hdf.format_data_name(dataname_vtu)

parameters.update({'dataname': {'vtu': dataname_vtu,
                                'hdf': dataname_hdf}})

# --------------------------------------------------------------------------- #
# Ellipse parameters

rot_center = np.array((0.5, 0.5))
ell_center = rot_center + np.array((lambda3*lambda1, 0))

parameters.update({'rot_center': rot_center,  # Center of rotation of ellipse
                   'ell_center': ell_center,  # Center of ellipse
                   'ell_radius': (lambda1, lambda2*lambda1),  # Ellipse Radii
                   'angular_vel': (2*(lambda1)**2)**-1,   # angular velocity (rad/s)
                   'theta_init': 0.,  # initial ellipse angle w.r.t 1st axe
                   })

# --------------------------------------------------------------------------- #
# Mesh parameters

parameters.update({'h_mesh': 0.008,  # mesh size w.r.t 1m (d.u.)
              })

# --------------------------------------------------------------------------- #
# TimeSerie parameters

parameters.update({'dt': 0.005,       # Time step
              'T': 30.,          # Final time
              'nb_export': 2,    # Number of time-steps between each vtu export
              })

# --------------------------------------------------------------------------- #
# Parameters for optimizaiton algorithms

parameters.update({'pen': 1e-9,        # Coefficient de penalisation volumique (~0)
              'eps_u': 1e-4,      # Test for convergence of the solution
              'eps_lambda': 1e-1,  # Tolerance on relaxed constraints
              'c_lambda': 1,  # Tolerance on relaxed constraints
              })
# --------------------------------------------------------------------------- #
# Fluid and solid materials parameters

parameters.update({'rho': 1.,         # masse volumique du fluide
              'rho_delta': 1e0,    # rho_solide - rho_fluide
              'nu': 1./lambda4,  # visco dynamique du fluide, nu = 1/Re
              'nu_delta': 1e0,     # nu_solide - nu_fluide
              })

# --------------------------------------------------------------------------- #
# POD data parameters
TMIN = 15.      # Time for the first snapshot
TMAX = 20.      # Time for the last snapshot

TEXPORT = parameters['nb_export']*parameters['dt']

NMIN = int(TMIN/TEXPORT)
NMAX = int(TMAX/TEXPORT)

DECIM = max((int((NMAX-NMIN)/500.), 1))

# options to control which snapshots are read
parameters['load'] = {'tmin': TMIN,       # starting time
                      'imin': NMIN,       # starting index
                      'tmax': TMAX,       # stoping time
                      'imax': NMAX,       # stoping index
                      'decim': DECIM      # read one snapshot every 'decim' snapshots
                      }

# --------------------------------------------------------------------------- #
# POD computation parameters

# Threshold error on eigen-values energy
POD_THRESHOLD = 1e-3

# Maximum number of modes
POD_NMODESMAX = 25
parameters['pod'] = {'thld': POD_THRESHOLD,
                     'nmax': POD_NMODESMAX}

# --------------------------------------------------------------------------- #
# ROM runtime parameters

# ROM Time step
DT_ROM = TEXPORT*parameters['load']['decim']

# ROM Stabilization coefficient for mode n: $mu_n = mu(1+stab*n)$
STAB_ROM = 0.

parameters['rom'] = {'dt': DT_ROM,
                     'stab': STAB_ROM
                     }

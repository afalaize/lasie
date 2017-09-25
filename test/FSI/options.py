#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:52:34 2017

@author: afalaize
"""
from lasie_rom.io import hdf
from main import parameters

options = {}

# --------------------------------------------------------------------------- #
# Name of the data to retrieve from the vtu files
if parameters['dataname'] is None:
    options['dataname'] = None
else:
    options['dataname'] = hdf.format_data_name(parameters['dataname'])

# --------------------------------------------------------------------------- #
# LOAD OPTIONS
TMIN = 15.      # Time for the first snapshot
TMAX = 20.      # Time for the last snapshot

TEXPORT = parameters['nb_export']*parameters['dt']

NMIN = int(TMIN/TEXPORT)
NMAX = int(TMAX/TEXPORT)

DECIM = max((int((NMAX-NMIN)/500.), 1))

# options to control which snapshots are read
options['load'] = {'imin': NMIN,       # starting index
                   'imax': NMAX,       # stoping index
                   'decim': DECIM      # read one snapshot every 'decim' snapshots
                   }

# --------------------------------------------------------------------------- #
# POD OPTIONS

# Threshold error on eigen-values energy
THRESHOLD = 1e-3

# Maximum number of modes
NMODESMAX = 50

options['pod'] = {'thld': THRESHOLD,
                  'nmax': NMODESMAX}

options['rom'] = {'dt': TEXPORT*options['load']['decim'],
                  'stab': 0.}


# --------------------------------------------------------------------------- #
# ROM OPTIONS
eps_u = 1e-6
eps_lambda = 10

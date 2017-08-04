# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:18:40 2017

@author: afalaize
"""

import tables  # Used in compression FILTER definition
import numpy as np


# define the compression method (filter) for hdf5 files
#FILTER = tables.Filters(complevel=5, complib='blosc')      # compression
FILTER = None                                               # no compression

# define the standard data type
DTYPE = np.dtype(float)

# Standard name for pvd file associated with Thost time-series
PVDNAME = 'ThostA.pvd'

# Order for array reshaping
ORDER = 'C'
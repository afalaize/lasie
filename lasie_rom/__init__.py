#
from __future__ import absolute_import
from . import pod, deim, io, parallelization, grids, operators

import os
path_to_package = os.path.realpath(__file__)[:os.path.realpath(__file__).rfind(os.sep)]

__all__ = ['pod', 'deim', 'io', 'parallelization',
           'grids', 'path_to_package', 'operators']

__author__ = "Antoine Falaize"
__version__ = "17.09.a"
__licence__ = \
    "CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)"
__author__ = "Antoine Falaize"
__maintainer__ = "Antoine Falaize"
__author_email__ = 'antoine.falaize@univ-lr.fr'
__copyright__ = "Copyright 2016-2017"

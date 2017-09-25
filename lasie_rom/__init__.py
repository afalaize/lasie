#
from __future__ import absolute_import

from . import pod
from . import classes
from . import deim
from . import io
from . import parallelization
from . import grids
from . import operators
from . import plots
from . import rom

import os
path_to_package = os.path.realpath(__file__)[:os.path.realpath(__file__).rfind(os.sep)]

__all__ = ['pod', 'deim', 'io', 'parallelization', 'classes',
           'grids', 'path_to_package', 'operators', 'plots', 'rom']

__author__ = "Antoine Falaize"
__version__ = "17.09.a"
__licence__ = \
    "CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)"
__author__ = "Antoine Falaize"
__maintainer__ = "Antoine Falaize"
__author_email__ = 'antoine.falaize@univ-lr.fr'
__copyright__ = "Copyright 2016-2017"

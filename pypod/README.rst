This is a python module for the order reduction of CFD models in the context
of the Thost software.

Content
=======

config.py
----------

This is where all configuration options are defined.

vtu2hdf.py
-----------

This is where we define the function for converting .vtu Thost results to hdf5 file format for python processing.

Informations
=============

The ordering of axes for every arrays is (nx, nt, nc1, nc2, nc3...) with

* nx: the number of nodes in the mesh
* nt: the number of time steps
* ncj: the number of components in the j-th data dimension

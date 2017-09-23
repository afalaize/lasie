A python package for model order reduction in computational fluid dynamics.


Installation
=============

Prerequisites
--------------
The following dependencies will be automatically installed with pip:

- `numpy <http://www.numpy.org>`_
- `matplotlib <http://matplotlib.org/>`_
- ̀`multiprocessing <https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing>`_
- `progressbar2 <https://pypi.python.org/pypi/progressbar2>`_
- `tables <http://www.pytables.org/index.html>`_
- `pyvtk <http://www.pytables.org/index.html>`_


Installation of vtk
--------------------

- on python 3.7: :code:`conda install vtk`
- on python 3.6: :code:`conda install -c clinicalgraphics vtk=7.1.0`

see:

https://stackoverflow.com/questions/43184009/install-vtk-with-anaconda-3-6


With pip
--------
The easiest way to install the package is via ``pip`` from the `PyPI (Python
Package Index) <https://pypi.python.org/pypi>`_::

    pip install pyphs

This includes the latest code and should install all dependencies automatically, except

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:17:57 2017

@author: afalaize
"""

from __future__ import absolute_import, print_function, division

import tables
import os
from . import vtk, hdf
from .. import parallelization as para


def vtk2hdf(vtk_path, hdf_path):
    """
    Read data from a .vtk (or .vtu) file and write them to a .hdf5 file.

    Parameters
    -----------
    vtk_path: str
        The path to the .vtk or .vtu file to read.

    hdf_path: str
        The path to the .hdf5 file where the data shall be written.

    data_names: list of str
        The list of data names to retrieve from the .vtu file.

    Output
    ------

    None

    Remark
    ------
    1. The mesh is systematically retrieved from the .vtu file.
    2. The (metadata) title of the hdf5 file is all the names of the data
        (plus the 'mesh'), separated with blank spaces.
    """

    # retrieve the elementTree from vtu (ie. xml) file
    data = vtk.read(vtk_path)

    # title for the hdf file: a list of data names recovered from the .vtk file
    hdf_title = ' '.join(list(map(hdf.format_data_name, data.keys())))

    # open the hdf file in write mode
    hdf_file = tables.open_file(hdf_path, mode='w', title=hdf_title)

    # a call to the write function for each name in data_names...
    for k in data.keys():
        hdf.write_array_in_hdf(hdf_file, data[k], k)

    # close the hdf file
    hdf_file.close()


def pvd2hdf(pvd_path, hdf_folder, imin=None, imax=None, decim=None,
            tmin=None, tmax=None):
    """
    Convert data from all .vtk files listed in a .pvd file to .hdf5 files.

    Parameters
    -----------
    pvd_path: str
        The path to the .pvd file that lists all .vtu files to convert.

    hdf_folder: str
        The folder where the .hdf5 files shall be written.

    imin: Int or None
        the index in the .pvd file of the first .vtu file to convert. If None,
        the start index is set to 0 (default is None).

    imax: Int or None
        the index in the .pvd file of the last .vtu file to convert. If None,
        the end index is set to the last file (default is None).

    decim: Int > 0 or None
        Decimation coefficient; e.g. with decim=2, one file over two is read.

    Output
    ------

    None

    Remark
    ------
    An additional text file with name 'listOfHdfFiles.txt' is created in
    the folder pointed by 'hdf_folder', to summarize the written hdf files,
    with corresponding time steps.
    """
    # Retrieve lists of .vtu files and asssociated times
    listOfVtuFiles = vtk.pvd2files(pvd_path)
    listOfTimes = vtk.pvd2times(pvd_path)

    # self explanatory
    assert len(listOfVtuFiles) == len(listOfTimes), \
        '{} != {}'.format(len(listOfVtuFiles), len(listOfTimes))

    if imin is None:
        imin = 0

    if imax is None:
        imax = len(listOfVtuFiles)

    if decim is None:
        decim = 1

    if not os.path.exists(hdf_folder):
        os.makedirs(hdf_folder)

    arguments = list()
    for i, vtu_path in enumerate(listOfVtuFiles):
        if (imin <= i < imax) and ((i-imin) % decim == 0):
            arguments.append((vtu_path, listOfTimes[i]))

    def myfunc(arg):
        vtu_path, current_time = arg
        hdf_path = hdf_folder + os.sep + vtu_path[:-3].split(os.sep)[-1] + 'hdf5'
        vtk2hdf(vtu_path, hdf_path)
        print('Converting {}'.format(vtu_path[:-3].split(os.sep)[-1]))

    para.map(myfunc, arguments)

    listOfHdfFiles = open(os.path.join(hdf_folder, 'listOfHdfFiles.txt'), 'w')
    for vtu_path, current_time in arguments:
        hdf_path = hdf_folder + os.sep + vtu_path[:-3].split(os.sep)[-1] + 'hdf5'
        listOfHdfFiles.write('{} {}\n'.format(current_time, hdf_path))
    listOfHdfFiles.close()

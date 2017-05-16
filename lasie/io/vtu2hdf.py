# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:17:57 2017

@author: afalaize
"""

from __future__ import absolute_import, print_function, division
import tables
import os
from .vtu.read import (getElementTree, getDataFromElementTree, 
                                     getCoordinatesFromElementTree, 
                                     pvd2ListOfFiles, pvd2ListOfTimes)

from .hdf.tools import format_data_name
from .hdf.write import writeText2Hdf
from ..parallelization.tools import parmap


def vtu2Hdf(vtu_path, hdf_path, data_names):
    """
    Read data from a .vtu file and write them to a .hdf5 file.
    
    Parameters
    -----------
    vtu_path: str
        The path to the .vtu file to read.
        
    hdf_path: str
        The path to the .hdf5 file where the data shall be written.
        
    data_names: list of str
        The list of data names to retrieve from the .vtu file.
        
    Output
    ------
    
    None
    
    Remark:
        1. The mesh is systematically retrieved from the .vtu file.
        2. The (metadata) title of the hdf5 file is all the names of the data 
            (plus the 'mesh'), separated with blank spaces.
    """
    
    # title for the hdf file: a list of data names recovered from the .vtu file
    hdf_title = 'mesh' + (' {}'*len(data_names)).format(*map(format_data_name,
                                                             data_names))
    # open the hdf file in write mode
    hdf_file = tables.open_file(hdf_path, mode='w', title=hdf_title)
    
    # retrieve the elementTree from vtu (ie. xml) file
    tree = getElementTree(vtu_path)
    
    # a call to the write function for each name in data_names...
    for data_name in data_names:
        # retrieve the data from vtu file DataArray with name 'data_name'
        text = getDataFromElementTree(tree, data_name=data_name)
        # write the data to an extendable array
        writeText2Hdf(hdf_file, text, data_name)
        
    # plus a call to the write function for the mesh
    mesh_text = getCoordinatesFromElementTree(tree)
    
    # the number of components (3) is not correct so we select first 2 values in each line
    new_mesh_text = "\n"
    for l in mesh_text.splitlines():
        elts = l.split(' ')[:2]
        try:
            new_line = ('{} '*2).format(*elts)[:-1] + '\n'
            new_mesh_text += new_line
        except IndexError:
            pass
    new_mesh_text = new_mesh_text[:-1]
    writeText2Hdf(hdf_file, new_mesh_text, 'mesh')

    # close the hdf file
    hdf_file.close()


def pvd2Hdf(pvd_path, hdf_folder, data_names,
            imin=None, imax=None, decim=None):
    """
    Convert data from all .vtu files listed in a .pvd file to .hdf5 files.
    
    Parameters
    -----------
    pvd_path: str
        The path to the .pvd file that lists all .vtu files to convert.
        
    hdf_folder: str
        The folder where the .hdf5 files shall be written.
        
    data_names: list of str
        The list of data names to retrieve from each .vtu file ('mesh' is
        automatically appended to that list).
        
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
    
    Remark:
        An additional text file with name 'listOfHdfFiles.txt' is created in 
        the folder pointed by 'hdf_folder', to summarize the written hdf files,
        with corresponding time steps.
    """
    # Retrieve lists of .vtu files and asssociated times
    listOfVtuFiles = pvd2ListOfFiles(pvd_path)
    listOfTimes = pvd2ListOfTimes(pvd_path)
    
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
        vtu2Hdf(vtu_path, hdf_path, data_names)
        print('Converting {}'.format(vtu_path[:-3].split(os.sep)[-1]))
        
    parmap(myfunc, arguments)
    
    listOfHdfFiles = open(hdf_folder + os.sep + 'listOfHdfFiles.txt' , 'w')
    for vtu_path, current_time in arguments:
        hdf_path = hdf_folder + os.sep + vtu_path[:-3].split(os.sep)[-1] + 'hdf5'
        listOfHdfFiles.write('{} {}\n'.format(current_time, hdf_path))
    listOfHdfFiles.close()     
    
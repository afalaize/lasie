# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:17:57 2017

@author: afalaize
"""

import tables
import os
import numpy as np
from .read_vtu import (getElementTree, getDataFromElementTree, 
                       getCoordinatesFromElementTree, 
                       pvd2ListOfFiles, pvd2ListOfTimes)
from ..config import DTYPE, FILTER
from .tools import getFolderFromPath


def format_data_name(data_name, replacement_char='_'):
    """
    Format data_name to comply with HDF5 file format constraints for data names
    Take everything before the first '(' and replace forbiden characters with 
    the replacement_char.
    
    Parameters
    -----------
    data_name: str
        the srtring to format
        
    replacement_char: single char
        The default character to replace each forbiden character (the 
        default is '_').
    
    Output
    -------
    formated_data_name: str
        The properly formated data name.
    """
    # List of forbiden chars to be removed from data_name
    forbiden_chars = (r'/', )
    
    # replace each forbiden char with the replacement char
    for char in forbiden_chars:
        data_name = data_name.replace(char, replacement_char)
        
    # Lower case data name
    data_name = data_name.lower()

    # take everything before the first '(' if any, else the whole string
    data_name = data_name.split('(')[0]
    
    # return formated data_name
    return data_name


def dumpArrays2Hdf(arrays, names, hdf_path):
    """
    write all array in (list of) arrays under the corresponding name in (list
    of) names to the hdf5 file with path 'hdf_path'.
    """
    # title for the hdf file: a list of data names recovered from the .vtu file
    hdf_title = (' {}'*len(names)).format(*map(format_data_name, names))[1:]
    # recover the folder from the hdf_path
    folder = getFolderFromPath(hdf_path)
    # create that fodler if it does not exist
    if folder is not None and not os.path.exists(folder):
        os.makedirs(folder)    
    # open the hdf file in write mode
    hdf_file = tables.open_file(hdf_path, mode='w', title=hdf_title)
    for array, name in zip(arrays, names):
        writeArray2Hdf(hdf_file, array, name)        
    hdf_file.close()    
    
    
def writeArray2Hdf(hdf_file, array, data_name):
    """
    a convenient function to actually write the data (numpy arrray) to the 
    opened hdf file.
    """

    # define the dtypes
    atom = tables.Atom.from_dtype(DTYPE)
    
    # get a sample to determine the shape of the data
    sample_data = array[0]

    sample_data_shape = sample_data.shape
    
    # define the shape of the data, with first (extendable) dimension set to 0
    shape = tuple([0, ] + list(sample_data_shape))
    
    # build the storage object, here an "extendable array" (earray)
    hdf_data_name = format_data_name(data_name)
    data_storage = hdf_file.create_earray(hdf_file.root, hdf_data_name, atom,
                                          shape=shape,
                                          filters=FILTER)

    # get the data from each text line and append the data list
    for data in array:
        data = data.reshape(sample_data_shape)
        # store the data in the hdf file
        data_storage.append(data[None])
        
    
def writeText2Hdf(hdf_file, text, data_name):
    """
    a convenient function to actually write the data (text format) to the 
    opened hdf file.
    """

    # define the dtypes
    atom = tables.Atom.from_dtype(DTYPE)
    
    # split text in lines, and remove the first and last lines
    text_lines = text.splitlines()[1:-1]

    # get a sample to determine the shape of the data
    sample_data = np.fromstring(text_lines[0], dtype=DTYPE, sep=' ')
    sample_data_shape = sample_data.shape

    # define the shape of the data, with first (extendable) dimension set to 0
    shape = tuple([0, ] + list(sample_data_shape))
    
    # build the storage object, here an "extendable array" (earray)
    hdf_data_name = format_data_name(data_name)
    data_storage = hdf_file.create_earray(hdf_file.root, hdf_data_name, atom,
                                          shape=shape,
                                          filters=FILTER)

    # get the data from each text line and append the data list
    for line in text_lines:
        # read the data
        data = np.fromstring(line, dtype=float, sep=' ').reshape(sample_data_shape)
        # store the data in the hdf file
        data_storage.append(data[None])
        
    
def vtu2Hdf(vtu_path, hdf_path, data_names, nc=3):
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
    
    # the number of components is not correct so we select first 2 values in each line
    new_mesh_text = ""
    for l in mesh_text.splitlines():
        elts = l.split(' ')[:2]
        try:
            new_line = ('{} '*nc).format(*elts)[:-1] + '\n'
            new_mesh_text += new_line
        except IndexError:
            pass
    new_mesh_text = new_mesh_text[:-1]
    writeText2Hdf(hdf_file, new_mesh_text, 'mesh')

    # close the hdf file
    hdf_file.close()


def pvd2Hdf(pvd_path, hdf_folder, data_names, nc=3, 
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
    
    listOfHdfFiles = open(hdf_folder + os.sep + 'listOfHdfFiles.txt' , 'w')
    for i, vtu_path in enumerate(listOfVtuFiles):
        if (imin <= i < imax) and ((i-imin) % decim == 0):
            print('Convert vtu to hdf5 {}/{}.'.format(i+1-imin, imax-imin))
            hdf_path = hdf_folder + os.sep + vtu_path[:-3].split(os.sep)[-1] + 'hdf5'
            vtu2Hdf(vtu_path, hdf_path, data_names, nc=nc)
            listOfHdfFiles.write('{} {}\n'.format(listOfTimes[i], hdf_path))

    listOfHdfFiles.close()     
    
    
if __name__ is '__main__':
    from config import PVDNAME
    from main import CONFIG
    pvd_path = CONFIG['vtu_folder'] + os.sep + PVDNAME
    pvd2Hdf(pvd_path, CONFIG['hdf_folder'], CONFIG['data_names_vtu'], 
            start=CONFIG['start'], end=CONFIG['end'])
    
    

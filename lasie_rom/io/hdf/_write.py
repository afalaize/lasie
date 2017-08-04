#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:42:37 2017

@author: root
"""

from __future__ import print_function, division, absolute_import

import os
import numpy as np
import tables
from .._tools import getFolderFromPath
from ._tools import format_data_name

from .. import config

def data2hdf(data, hdf_path):
    """
    write all arrays in dictionary of data to corresponding name in the hdf5
    file with path 'hdf_path'.
    """
    # title for the hdf file: a list of data names recovered from the .vtu file
    hdf_title = ' '.join(list(map(format_data_name, data.keys())))
    # recover the folder from the hdf_path
    folder = getFolderFromPath(hdf_path)
    # create that fodler if it does not exist
    if folder is not None and not os.path.exists(folder):
        os.makedirs(folder)
    # open the hdf file in write mode
    hdf_file = tables.open_file(hdf_path, mode='w', title=hdf_title)
    for k in data.keys():
        write_array_in_hdf(hdf_file, data[k], k)
    hdf_file.close()


def write_array_in_hdf(hdf_file, array, data_name):
    """
    a convenient function to actually write data (numpy arrray) to the
    opened hdf file.
    """

    # define the dtypes
    atom = tables.Atom.from_dtype(config.DTYPE)

    # get a sample to determine the shape of the data
    sample_data = array[0]

    sample_data_shape = sample_data.shape

    # define the shape of the data, with first (extendable) dimension set to 0
    shape = tuple([0, ] + list(sample_data_shape))

    # build the storage object, here an "extendable array" (earray)
    hdf_data_name = format_data_name(data_name)
    data_storage = hdf_file.create_earray(hdf_file.root, hdf_data_name, atom,
                                          shape=shape,
                                          filters=config.FILTER)

    # get the data from each text line and append the data list
    for data in array:
        data = data.reshape(sample_data_shape)
        # store the data in the hdf file
        data_storage.append(data[None])


def write_text_in_hdf(hdf_file, text, data_name):
    """
    a convenient function to actually write data (text format) to the
    opened hdf file.
    """

    # define the dtypes
    atom = tables.Atom.from_dtype(config.DTYPE)

    # split text in lines, and remove the first and last lines
    text_lines = text.splitlines()[1:-1]

    # get a sample to determine the shape of the data
    sample_data = np.fromstring(text_lines[0], dtype=config.DTYPE, sep=' ')
    sample_data_shape = sample_data.shape

    # define the shape of the data, with first (extendable) dimension set to 0
    shape = tuple([0, ] + list(sample_data_shape))

    # build the storage object, here an "extendable array" (earray)
    hdf_data_name = format_data_name(data_name)
    data_storage = hdf_file.create_earray(hdf_file.root, hdf_data_name, atom,
                                          shape=shape,
                                          filters=config.FILTER)

    # get the data from each text line and append the data list
    for line in text_lines:
        # read the data
        data = np.fromstring(line, dtype=float, sep=' ').reshape(sample_data_shape)
        # store the data in the hdf file
        data_storage.append(data[None])


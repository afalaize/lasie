# -*- coding: utf-8 -*-
"""

We make use the xml.etree module.

Created on Mon Feb 27 14:36:11 2017

@author: afalaize

"""

#                             !!! Warning !!!
# This module was firstly intended for reading of vtk files, that are basically
# xml files. Now, propper vtk reader is available, so we just let this script
# as a bunch of routines for reading xml files, but a lot of links may be
# broken, so use with care.
















































from __future__ import absolute_import

import numpy as np
from xml.etree import cElementTree as ElementTree


def read_xml(path, tag):
    """
read_xml
********

Read .xml file and extract the data with specified tag.

Parameters
----------

path : str
    Path to the .vtu file to read.

name : str
    Tag of the to retrieve in the .vtu file

Return
------

data : numpy array
    The data with specified tag from the .vtu file located at path.
    """
    tree = getElementTree(path)
    text = getDataFromElementTree(tree, tag)
    array = text2array(text)
    return array


def getElementTree(path):
    """
    get the root ElementTree from the .vtu file with path "vtu_path".
    """
    vtu_file = open(path, mode='r')
    tree = ElementTree.parse(vtu_file)
    vtu_file.close()
    return tree


def getDataFromElementTree(tree, dataname):
    """
    Extract the data with tag "dataname" in xml tree and return
    the associated text (string).

    Usage
    ------
    data_text = getDataFromElementTree(tree, data_name)

    Parameters
    ----------
    tree: ElementTree.tree
        An ElementTree.tree (see function getElementTree).

    dataname: raw str
        The label of the data to extract from the ElementTree.

    Return
    -------
    text: string
        The data with tag "dataname" in file "filename" as a multilines string.
    """

    # Init output
    text = None

    # list all elements in the xml tree with tag DataArray
    for elem in tree.getiterator('DataArray'):
        # if the name of data is dataname
        if 'Name' in elem.attrib and elem.attrib['Name'] == dataname:
            # update text output with the xml element text
            text = elem.text

    # raise an error if text is still None
    assert text is not None, \
        'Can not find tag "{}" in xml tree.'.format(dataname)

    return text


def getCoordinatesFromElementTree(tree):
    """
    Return the coordinates of mesh points from xml tree (ElementTree) as text.

    Usage
    ------
    data_text = getCoordinatesFromElementTree(tree)

    Parameters
    ----------
    tree: ElementTree.tree
        An ElementTree.tree (see function getElementTree here).

    Return
    -------
    text: string
        The mesh as a multilines string.
    """
    return tree.getiterator('Points')[0][0].text


def text2array(text):
    """
    Format the data from text (string) to a 2D numpy.array.
    """
    # init the data as a list
    data = list()

    # get the data from each text line and append the data list
    for line in text.splitlines()[1:-1]:
        data.append(np.fromstring(line, dtype=float, sep=' '))

    # transform the list of 1D array to a 2D array and return
    return np.array(data)

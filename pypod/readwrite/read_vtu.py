# -*- coding: utf-8 -*-
"""

Here, we define a set of functions for reading from '.vty' files (legacy xml).
We make use the xml.etree module.

Created on Mon Feb 27 14:36:11 2017

@author: afalaize

"""

from __future__ import absolute_import


import os
import numpy as np
from xml.etree import cElementTree as ElementTree
from collections import OrderedDict


def getElementTree(vtu_path):
    """
    get the root ElementTree from the .vtu file with path "vtu_path". 
    """
    vtu_file = open(vtu_path, mode='r')
    tree = ElementTree.parse(vtu_file)
    vtu_file.close()
    return tree

    
def getDataFromElementTree(tree, data_name):
    """
    Extract the data in xml file "filename" with tag "dataname" and return 
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
        if elem.attrib.has_key('Name') and elem.attrib['Name'] == data_name:
            # update text output with the xml element text
            text = elem.text
        
    # raise an error if text is still None (no DataArray with Name='dataname' found)
    assert text is not None, 'Can not find tag \n{} in xml tree.'.format(data_name)
    
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


def pvd2ListOfFiles(pvd_path):
    """
    Extract a list of .vtu file paths from the .pvd file pointed by pvd_path.
    
    Warning
    -------
    Duplicate .vtu file paths are deleted, so the list can possibly not contain
    the whole list of paths.
    """
    pvdtree = ElementTree.parse(open(pvd_path))
    folder = pvd_path[:pvd_path.rfind(os.sep)]
    listOfFiles = list()
    for elem in pvdtree.getiterator('DataSet'):
        path = elem.attrib['file']
        istartname = path.rfind(os.sep)
        if istartname == -1:
            istartname = 0
        else:
            istartname += 1
        listOfFiles.append(folder + os.sep + path[istartname:])
    # WARNING: we remove duplicate filenames
    return list(OrderedDict.fromkeys(listOfFiles))


def pvd2ListOfTimes(pvd_path):
    """
    Extract a list of times from the .pvd file pointed by pvd_path.

    Warning
    --------
    Duplicate time values are deleted, so the list can possibly not contain
    the whole list of time values.
    """
    pvdtree = ElementTree.parse(open(pvd_path))
    listOfTimes = list()
    for elem in pvdtree.getiterator('DataSet'):
        listOfTimes.append(float(elem.attrib['timestep']))
    # WARNING: we remove duplicate times
    return list(OrderedDict.fromkeys(listOfTimes))

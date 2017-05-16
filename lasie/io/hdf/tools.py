#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:05:57 2017

@author: root
"""

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
    _data_name = data_name.lower()

    # take everything before the first '(' if any, else the whole string
    _data_name = _data_name.split('(')[0]
    
    # return formated data_name
    return _data_name

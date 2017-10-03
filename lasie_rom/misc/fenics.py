#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def replace_pos_symbs_by_coord_vec_elements(ccode, symb='x', dim=2):
    """
    Replace occurences of "xi" with "x[i]" for i in range(dim) and the symbol
    "x" specified by symb.

    Parameters
    ----------

    ccode: str
        String associated with a piece of C code.

    symb: str
        String to search for in ccode. E.g. with symb='toto', occurences of
        'totoi' with i an integer are replaced with 'toto[i]'.

    dim: int
        The number of components of vector represented by 'symb'.

    Example
    -------

    >>> ccode = 'pow(pow(x0, 2)+pow(x1, 2), 0.5)'
    >>> replace_pos_symbs_by_coord_vec_elements(ccode, symb='x', dim=2)
    'pow(pow(x[0], 2)+pow(x[0], 2), 0.5)'
    """
    if isinstance(ccode, list):
        for i, e in enumerate(ccode):
            ccode[i] = replace_pos_symbs_by_coord_vec_elements(e,
                                                               symb=symb,
                                                               dim=dim)
    elif isinstance(ccode, str):
        for i in range(dim):
            old = symb+str(i)
            new = 'x[{0}]'.format(i)
            ccode = ccode.replace(old, new)
    else:
        raise TypeError('Unknown type for ccode.')
    return ccode

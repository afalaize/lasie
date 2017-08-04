
# coding: utf-8

# In[ ]:

from scipy.interpolate import interp1d

import scipy.linalg as lg

import numpy as np


###############################################################################


def IS(p, M, mu, r, kind='slinear'):
    """
    p : echantillon de parametres (list de floats)
    M : echantillon de bases (list de arrays)
    mu : nouveau paramètre (float)
    r : indice du point de reference dans M (int)
    kind : Interpolation method (str), see help on scipy.interpolate.interp1d
    """

    sigma = list()

    for i in range(len(p)):

        p1 = np.linalg.inv(np.dot(M[r].T, M[i]))

        U, s, V = lg.svd(np.dot(M[i], p1)-M[r])

        arctan_s = np.zeros((len(U[0]), len(V[:, 0])))
        for k in range(len(V[:, 0])):
            arctan_s[k, k] = np.arctan(s[k])

        sigma_i = np.dot(U, np.dot(arctan_s, V))
        sigma.append(sigma_i)
    interpolator = interp1d(p, sigma, axis=0, kind=kind)
    sigma_mu = interpolator(mu)
    ##################################################

    UU, ss, VV = lg.svd(sigma_mu)

    cos_SS = np.zeros((len(VV[0]), len(VV[:, 0])))
    for i in range(len(VV[:, 0])):
        cos_SS[i, i] = np.cos(ss[i])
    sin_SS = np.zeros((len(UU[0]), len(VV[:, 0])))
    for i in range(len(VV[:, 0])):
        sin_SS[i, i] = np.sin(ss[i])
    pp1 = np.dot(VV.T, cos_SS[0:len(VV[:, 0])])
    gamma1 = np.dot(np.dot(M[r], pp1) + np.dot(UU, sin_SS), VV)
    return gamma1

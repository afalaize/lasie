# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""


import sympy as sy
from lasie_rom.misc import tools


def build_Levelset_Sympy_Expression(ell_center, ell_radius, rot_center):
    """
    Return the sympy expression for the ealuation of the ellipse level-set function.
    """
    l = sy.symbols('l:2', real=True, positive=True)  # axis half-length
    a = sy.symbols('a:2', real=True)  #
    b = sy.symbols('b:2', real=True)
    x = sy.symbols('x:2', real=True)
    theta_symb = sy.symbols('theta', real=True, positive=True)

    D = sy.Matrix([[1./l[0]**2., 0.],
                   [0., 1./l[1]**2.]])

    R = sy.Matrix([[sy.cos(theta_symb), -sy.sin(theta_symb)],
                   [sy.sin(theta_symb), sy.cos(theta_symb)]])

    A = R*D*R.T

    v = (sy.eye(2)-R.T)*sy.Matrix(b)-sy.Matrix(a)
    c = 2.*R*D*v

    d = (v.T*D*v)[0, 0] - 1.

    Q = (sy.Matrix(x).T.dot(A.dot(sy.Matrix(x))))+c.T.dot(sy.Matrix(x))+d

    Q = Q.subs(dict([(ai, vali) for (ai, vali) in zip(a, ell_center)]))
    Q = Q.subs(dict([(bi, vali) for (bi, vali) in zip(b, rot_center)]))
    Q = Q.subs(dict([(ri, vali) for (ri, vali) in zip(l, ell_radius)]))

    return -Q


def build_Levelset_Dolfin_Expression(ell_center, ell_radius, rot_center):
    """
    Return a function that returns the dolfin expression for the level-set
    function associated with the ellipse.

    Parameters
    ----------

    ell_center : list of floats
        Coordinates (x1, x2) of the center of the ellipse.

    ell_radius : list of floats
        Ellipse axis half-length.

    rot_center : list of floats
        Coordinates (x1, x2) of the center of the rotation frame.

    Return
    ------

    f : function
        A function with parameters 'theta' (rotation angle) and 'degree' (that
        must correspond to the degree of the finite element function space),
        and that returns the dolfin expression.
    """
    import dolfin

    def levelset_Ccode():
        """
        Return the c code for the ealuation of the ellipse level-set function.
        """

        levelset = build_Levelset_Sympy_Expression(ell_center,
                                                   ell_radius,
                                                   rot_center)

        ccode = sy.printing.ccode(levelset)

        return tools.replace_pos_symbs_by_coord_vec_elements(ccode)

    levelset_c = levelset_Ccode()

    def levelset_Expression(theta, degree):
        """
        return a Dolfin Expression associated with the ellipse levelset
        function.
        """

        return dolfin.Expression(levelset_c, degree=degree, theta=theta)

    return levelset_Expression

# --------------------------------------------------------------------------- #

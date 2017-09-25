# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import dolfin
import os

import numpy as np
import matplotlib.pyplot as plt
import sympy as sy


# --------------------------------------------------------------------------- #

def sign_eps(x, eps=0.0):
    """
    eps can be regarded as an immersion precision (used iif mode=='tanh')
    """
    if eps == 0.0:
        return sy.sign(x)
    elif eps > 0.0:
        return sy.tanh(np.pi*x/eps)
    else:
        text = 'Immersion precision should be > 0 (got {})'.format(eps)
        raise AttributeError(text)


def heaviside_eps(x, eps=0.0):
    """
    eps can be regarded as an immersion precision (used iif mode=='tanh')
    """
    return 0.5*(1.+sign_eps(x, eps=eps))


# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #


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


def build_lambdified_levelset(ell_center, ell_radius, rot_center):
    levelset = build_Levelset_Sympy_Expression(ell_center, ell_radius,
                                               rot_center)
    x = sy.symbols('x:2', real=True)
    theta_symb = sy.symbols('theta', real=True, positive=True)
    return sy.lambdify([theta_symb, ] + list(x), levelset, modules='numpy')


def build_Levelset_Expression(ell_center, ell_radius, rot_center):
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

    def levelset_Ccode():
        """
        Return the c code for the ealuation of the ellipse level-set function.
        """

        levelset = build_Levelset_Sympy_Expression(ell_center,
                                                   ell_radius,
                                                   rot_center)

        ccode = sy.printing.ccode(levelset)

        return replace_pos_symbs_by_coord_vec_elements(ccode)

    levelset_c = levelset_Ccode()

    def levelset_Expression(theta, degree):
        """
        return a Dolfin Expression associated with the ellipse levelset
        function.
        """
        return dolfin.Expression(levelset_c, degree=degree, theta=theta)

    return levelset_Expression

# --------------------------------------------------------------------------- #


def build_fnchar_Expression(ell_center, ell_radius, rot_center, eps):
    """
    Return a function that returns the dolfin expression for the indicator
    function associated with en ellipse.

    Parameters
    ----------

    ell_center : list of floats
        Coordinates (x1, x2) of the center of the ellipse.

    ell_radius : list of floats
        Ellipse axis half-length.

    rot_center : list of floats
        Coordinates (x1, x2) of the center of the rotation frame.

    eps : float >= 0.
        Immersion precision

    Return
    ------

    f : function
        A function with parameters 'theta' (rotation angle) and 'degree' (that
        must correspond to the degree of the finite element function space),
        and that returns the dolfin expression.
    """
    def fnchar_Ccode():
        """
        Return the c code for the ealuation of the ellipse characteristic
        function.
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

        fnchar = heaviside_eps(-Q, eps=eps)

        ccode = sy.printing.ccode(fnchar)

        return replace_pos_symbs_by_coord_vec_elements(ccode)

    fnchar_c = fnchar_Ccode()

    def fnchar_Expression(theta, degree):
        """
        return a Dolfin Expression associated with the ellipse characteristic
        function.
        """
        return dolfin.Expression(fnchar_c, degree=degree, theta=theta)

    return fnchar_Expression
# --------------------------------------------------------------------------- #


def build_angle_Expression(center):
    """
    Return a function that returns the dolfin expression for the angle
    associated with polar coordinates.

    Parameter
    ---------

    center : list of floats
        Coordinates (x1, x2) of the center of the frame.

    """
    def angle_Ccode():
        x = sy.symbols('x:2', real=True)
        b = sy.symbols('b:2', real=True)
        theta = sy.atan2((x[1]-b[1]), (x[0]-b[0]))
        theta = theta.subs(dict([(bi, vali) for (bi, vali) in zip(b, center)]))
        ccode = sy.printing.ccode(theta)
        ccode = replace_pos_symbs_by_coord_vec_elements(ccode)
        return ccode

    angle_c = angle_Ccode()

    def angle_Expression(degree):
        return dolfin.Expression(angle_c, degree)

    return angle_Expression
# --------------------------------------------------------------------------- #


def build_velocity_Expression(center, angular_vel):
    """
    Return a function that returns the dolfin expression for the anglular
    velocity.

    Parameters
    ---------

    center : list of floats
        Coordinates (x1, x2) of the center of the rotating frame.

    angular_vel : float
        Magnitude of the angular velocity.

    """
    def velocity_Ccode():
        x = sy.symbols('x:2', real=True)
        b = sy.symbols('b:2', real=True)
        omega = sy.symbols('omega', real=True, positive=True)
        theta = sy.atan2((x[1]-b[1]), (x[0]-b[0]))
        x_vec = sy.Matrix(x)
        b_vec = sy.Matrix(b)
        dummy_vec = sy.Matrix([-sy.sin(theta), sy.cos(theta)])
        velocity_vec = omega*sy.sqrt((x_vec-b_vec).dot(x_vec-b_vec))*dummy_vec
        subs = dict([(bi, vali) for (bi, vali) in zip(b, center)])
        subs.update({omega: angular_vel})
        velocity_vec = velocity_vec.subs(subs)
        velocity = list(velocity_vec)
        ccode = [sy.printing.ccode(vi) for vi in velocity]
        ccode = replace_pos_symbs_by_coord_vec_elements(ccode)
        return tuple(ccode)

    velocity_c = velocity_Ccode()

    def velocity_Expression(degree):
        return dolfin.Expression(velocity_c, degree=degree)

    return velocity_Expression

# --------------------------------------------------------------------------- #

if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    mesh = dolfin.UnitSquareMesh(500, 500)

    # ----------------------------------------------------------------------- #

    degree = 1

    S = dolfin.FunctionSpace(mesh, "Lagrange", degree)   # Scalar
    fnchar = dolfin.Function(S)
    angle = dolfin.Function(S)

    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", degree)  # Vector
    velocity = dolfin.Function(V)

    # ----------------------------------------------------------------------- #

    # Center of rotation
    rot_center = np.array((0.5, 0.5))

    # Parameters
    lambda1 = 0.375  # Principal radius (m)
    lambda2 = .25  # Shape parameter (d.u.)
    lambda3 = 0.     # excentricity parameter (d.u.)
    lambda4 = 10.     # angular velocity (rad/s)
    eps = 0.5

    # Center of ellipse
    ell_center = rot_center + np.array((lambda3*lambda1, 0))

    # Radius in each dimensions
    ell_radius = lambda1, lambda2*lambda1

    angular_vel = lambda4

    v_expr = build_velocity_Expression(rot_center, angular_vel)(degree)
    fnchar_expr = build_fnchar_Expression(ell_center,
                                          ell_radius,
                                          rot_center,
                                          eps)(0.5, degree)

    levelset_expr = build_Levelset_Expression(ell_center,
                                              ell_radius,
                                              rot_center)(0.5, degree)

    file1 = dolfin.File("file1.pvd")
    file2 = dolfin.File("file2.pvd")
    file3 = dolfin.File("file3.pvd")
    file4 = dolfin.File("file4.pvd")

    plt.close('all')

#    # plt.figure()
#    # dolfin.plot(dolfin.project(v_expr, V=V))
#    file1 << dolfin.project(v_expr, V=V)
#
    # plt.figure()
    # dolfin.plot(dolfin.project(fnchar_expr, V=S))
    file2 << dolfin.project(fnchar_expr, V=S)

#    # plt.figure()
#    # dolfin.plot(dolfin.project(v_expr*fnchar_expr, V=V))
#    file3 << dolfin.project(v_expr*fnchar_expr, V=V)
#
#    # plt.figure()
#    # dolfin.plot(dolfin.project(v_expr*fnchar_expr, V=V))
#    file4 << dolfin.project(levelset_expr, V=S)

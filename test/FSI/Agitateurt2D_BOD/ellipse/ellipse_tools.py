# -*- coding: utf-8 -*-

import sympy as sy
from lasie_rom.misc import tools


def build_angle_Expression(center):
    """
    Return a function that returns the dolfin expression for the angle
    associated with polar coordinates.

    Parameter
    ---------

    center : list of floats
        Coordinates (x1, x2) of the center of the frame.

    """
    import dolfin

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
    import dolfin

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
        ccode = tools.replace_pos_symbs_by_coord_vec_elements(ccode)
        return tuple(ccode)

    velocity_c = velocity_Ccode()

    def velocity_Expression(degree):
        return dolfin.Expression(velocity_c, degree=degree)

    return velocity_Expression

# --------------------------------------------------------------------------- #

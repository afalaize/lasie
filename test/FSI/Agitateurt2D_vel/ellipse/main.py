# -*- coding: utf-8 -*-
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

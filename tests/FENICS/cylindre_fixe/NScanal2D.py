"""
ecoulement dans un canal avec un obstacle cylindrique en 2D

# h = 10     hauteur du canal
# l = 40     longueur du canal
# x = 10     centre du cylindre suivant x
# y = 0      centre du cylindre suivant y
# r = 0.5    rayon du cylindre

"""

# ----------------------------PYTHON 3 COMPATIBILITY ------------------------ #
from __future__ import division, absolute_import, print_function

# -----------------------------  IMPORT OTHERS ------------------------------ #
import os

# -----------------------------  IMPORT DOLFIN ------------------------------ #
import dolfin

# Print log messages only from the root process in parallel
dolfin.parameters["std_out_all_processes"] = False;


# ------------------------  IO Parameters  ------------------------------------

here = os.path.realpath(__file__)[:os.path.realpath(__file__).rfind(os.sep)]
res_folder = os.path.join(here, 'RESULTS')

if not os.path.exists(res_folder):
    os.mkdir(res_folder)                

# fichiers de sauvergarde
# os.path.join returns unicode string so that filenames ...
# ... have to be converted back to regular strings
ufile = dolfin.File(str(os.path.join(res_folder, 'velocity.pvd')))
pfile = dolfin.File(str(os.path.join(res_folder, 'pressure.pvd')))

# -----------------------  PHYSICAL PARAMETERS ------------------------------ #
rho = 1       # masse volumique
dt = 0.01     # pas de temps
T = 2000      # temps final
nu = 0.01     # visco dynamique   Re = rho*u*d/nu ici rho=d=u=1 ---> nu = 1/Re


# --------------------------------  MESH ------------------------------------ #

# in command line, use dolfin-convert canal2D.msh canal2D.xml
# to generate canal2D.xml, canal2D_facet_region.xml and canal2D_physical_region.xml


mesh = dolfin.Mesh("maillage/canal2D.xml") 
sub_domains = dolfin.MeshFunction("size_t", mesh, 
                                  "maillage/canal2D_facet_region.xml") 


# ----------------------------- FUNCTIONAL SPACES --------------------------- #
V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 2)   # vitesse
Q = dolfin.FunctionSpace(mesh, "Lagrange", 1)         # pression


#%% -------------------------- CAUCHY STRESS TENSOR ------------------------- #
def sigma(v, w):
    """
Cauchy stress tensor

Parameters
----------
v: element from dolfin.VectorFunctionSpace
    Velocity trial function.
w: element from dolfin.FunctionSpace
    Pressure trial function.
    
Returns
-------
sigma: dolfin function
    The cauchy stress tensor associated with :code:`v` and  :code:`w`:
    :math:`\\nu \\left( \\nabla v + \\nabla v^\\intercal \\right) - w \\mathbf I` 
    """
    return (nu*(dolfin.grad(v) + dolfin.grad(v).T) - 
            w*dolfin.Identity(v.geometric_dimension()))

#%% Conditions limites 
intflow = dolfin.DirichletBC(V, ("1","0.0"), sub_domains, 31)    # u=1 v=0
paroies = dolfin.DirichletBC(V.sub(1), 0, sub_domains,33)        # v=0 paroie glissante
cylindre = dolfin.DirichletBC(V, (0,0), sub_domains, 35)         # u=v=0 noslip 
outflow = dolfin.DirichletBC(Q, 0, sub_domains, 34)              # p=0
bcu = [intflow, cylindre, paroies]                               # conditions en vitesse
bcp = [outflow]                                                  # conditions en pression

# Trials and tests functions
u = dolfin.TrialFunction(V)
p = dolfin.TrialFunction(Q)
v = dolfin.TestFunction(V)
q = dolfin.TestFunction(Q)

# Functions
u0 = dolfin.Function(V)
p0 = dolfin.Function(Q)
u1 = dolfin.Function(V)
p1 = dolfin.Function(Q)
f = dolfin.Constant((0, 0))


# Tentative velocity step
F1 = (rho/dt)*dolfin.inner(u - u0, v)*dolfin.dx + rho*dolfin.inner(dolfin.grad(u0)*u0, v)*dolfin.dx + \
     nu*dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dx - dolfin.inner(f, v)*dolfin.dx
a1 = dolfin.lhs(F1)
L1 = dolfin.rhs(F1)

# Pressure update
a2 = dolfin.inner(dolfin.grad(p), dolfin.grad(q))*dolfin.dx
L2 = -(rho/dt)*dolfin.div(u1)*q*dolfin.dx

# Velocity update
a3 = dolfin.inner(u, v)*dolfin.dx
L3 = dolfin.inner(u1, v)*dolfin.dx - (dt/rho)*dolfin.inner(dolfin.grad(p1), v)*dolfin.dx

# Assemble matrices
A1 = dolfin.assemble(a1)
A2 = dolfin.assemble(a2)
A3 = dolfin.assemble(a3)

# Use amg preconditioner if available
prec = "amg" if dolfin.has_krylov_solver_preconditioner("amg") else "default"

valx=list()
valy=list()
energy=list()

nb=0
t = dt
while t < T :

   # Save to fileimport dolfin_utils
    if nb < 100:
        nb += 1
    else:
        ufile << u0
        pfile << p0
        nb = 0
  
    # Compute tentative velocity step 
    dolfin.begin("Computing tentative velocity")
    b1 = dolfin.assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    dolfin.solve(A1, u1.vector(), b1, "gmres", "default")
    dolfin.end()

    # Pressure correction
    dolfin.begin("Computing pressure correction")
    b2 = dolfin.assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    dolfin.solve(A2, p1.vector(), b2, "cg", prec)
    dolfin.end()

    # Velocity correction
    dolfin.begin("Computing velocity correction")
    b3 = dolfin.assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    dolfin.solve(A3, u1.vector(), b3, "gmres", "default")
    dolfin.end()
    
    #plot solution
    # dolfin.plot(p1, title="Pressure", rescale=True)
    # dolfin.plot(u1, title="Velocity", rescale=True)     
      
    # Forces sur le cylindre
    valx.append((t,dolfin.assemble(dolfin.dot(dolfin.FacetNormal(mesh),sigma(u0,p0))[0]*dolfin.ds(35, domain=mesh, subdomain_data=sub_domains))))
    valy.append((t,dolfin.assemble(dolfin.dot(dolfin.FacetNormal(mesh),sigma(u0,p0))[1]*dolfin.ds(35, domain=mesh, subdomain_data=sub_domains))))
    energy.append((t,dolfin.assemble(dolfin.dot(u0,u0)*dolfin.dx)))
    
    # Move to next time step
    u0.assign(u1)
    t += dt
    print("{:2.2f}% done".format((t/T)*100))


# Save forces
#np.savetxt("results_%s/energy" %(fic),energy )
#np.savetxt("results_%s/fx" %(fic),valx )
#np.savetxt("results_%s/fy" %(fic),valy )    

# Plot solution
dolfin.plot(p1, title="Pressure", rescale=True)
dolfin.plot(u1, title="Velocity", rescale=True)
dolfin.interactive()


# Move mesh
#mesh = mesh.move(cylindre)


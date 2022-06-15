from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

from problems import Problem
import matplotlib.pyplot as plt
import numpy as np

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;




# Create mesh
Nx=50
problem_name = "cylinder"#"lid-driven_cavity"#"lid-driven_cavity"
physical_problem = Problem(problem_name, Nx)
mesh = physical_problem.mesh

# Set parameter values
Re= Constant(1000. )
nu = Constant(1./Re)
f = Constant((0., 0.))
u_top = Constant(1.)


degree_poly=2
scalar_element = FiniteElement("CG", mesh.ufl_cell(), degree_poly)
vector_element = VectorElement("CG", mesh.ufl_cell(), degree_poly)
scalar_element_proj = FiniteElement("CG", mesh.ufl_cell(), degree_poly-1)
vector_element_proj = VectorElement("CG", mesh.ufl_cell(), degree_poly-1)
system_element = MixedElement( vector_element , scalar_element )
system_element_proj = MixedElement( vector_element_proj , scalar_element_proj )
dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
W = FunctionSpace(mesh,system_element)
W_proj = FunctionSpace(mesh,system_element_proj)
V = FunctionSpace(mesh,vector_element)
V_proj = FunctionSpace(mesh,vector_element_proj)




bcs = physical_problem.define_bc(W, u_top)


# Define trial and test functions

up = Function(W)
(u, p) = split(up)

uptrial = TrialFunction(W)
(utrial, vtrial) = split(uptrial)


vq = TestFunction(W)
(v, q)  = split(vq)


scalar_prod_form = inner(uptrial,vq)*dx
scalar_prod_matrix = assemble(scalar_prod_form)


# Define the forms
h = function.specialfunctions.CellDiameter(mesh)
hmin = mesh.hmin()

def dual_projection(z):
    z1=Function(W)
    z3=Function(W)
    scalar_prod_matrix.transpmult(z.vector(), z1.vector())  # z1 = A'*z
    z2=interpolate(interpolate(z1, W_proj),W)
    solve(scalar_prod_matrix, z3.vector(), z2.vector())  # solve A*z3 = z2
    return z3
    

def sigma_star(v):
    return v#-project(project(v,V_proj),V)

c1 = Constant(1.)
c2 = Constant(1.)
nu_local = nu

b_form = 0.5*(inner(dot(u,nabla_grad(u)),v)  + inner(dot(u,nabla_grad(v)),u) )*dx
a_form = 2*nu*inner(sym(grad(u)),sym(grad(v)))*dx

tau_den = c1*(nu+nu_local)/(h/degree_poly)**2+c2*project(sqrt(u[0]**2+u[1]**2),V0)/(h/degree_poly)
tau    = project(1./tau_den,V0)
s_conv = (tau*inner(sigma_star(dot(u,nabla_grad(u))),sigma_star(dot(u,nabla_grad(v))))) *dx
s_pres = (tau*inner(sigma_star(grad(p)),sigma_star(grad(q)))) *dx

F = b_form + a_form - inner(p,div(v))*dx+ s_conv + inner(div(u),q)*dx + s_pres

J = derivative(F, up)


# Prepare nonlinear solver
snes_solver_parameters = {"nonlinear_solver":"snes",
                          "snes_solver": {"method" : "anderson",
                                          "linear_solver": "mumps",
                                          "maximum_residual_evaluations":10000,
                                          "maximum_iterations": 1000,
                                          "report": True,
                                          "error_on_nonconvergence": False}
                          }

# Rey = 10000
# Newton by FEniCS not converging
# SNES solvers:
# newtonls not converging (super slow)
# nrichardson diverges
# qn diverges
# ncg diverges (fast)
# ngmres slowly slowly converging? I stopped at 1000 iter 1.e-04 
# fas segmentation fault
# nasm error
# aspin error
# ngs error
# anderson fast but sloooowly converging. 1000 iter 1.e-04
# ms too slow, not seeing anything
# composite segmentation fault error

problem = NonlinearVariationalProblem(F, up, bcs, J)
solver  = NonlinearVariationalSolver(problem)
#solver.parameters.update(snes_solver_parameters)

# Export the initial solution (zero)
outfile_u = File(physical_problem.name+"_steady/u.pvd")
outfile_p = File(physical_problem.name+"_steady/p.pvd")
outfile_ld = File(physical_problem.name+"_steady/ld.pvd")

solver.solve()



# Plot

(u, p) = up.split()
outfile_u << u
outfile_p << p

plt.figure();
pp=plot(p); plt.colorbar(pp);
plt.title("Pressure");
plt.show(block=False)

plt.figure()
pp=plot(u[0]); plt.colorbar(pp)
plt.title("u")
plt.show(block=False)

plt.figure()
pp=plot(u[1]); plt.colorbar(pp)
plt.title("v")
plt.show(block=False)


plt.figure()
pp=plot(u); plt.colorbar(pp)
plt.title("Velocity")
plt.show()

from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

import matplotlib.pyplot as plt

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;



# Set parameter values
Re= Constant(3000. )
nu = Constant(1./Re)
f = Constant((0., 0.))
u_top = Constant(1.)

# Create mesh
square = Rectangle(Point(0., 0.), Point(1., 1.))

Nx=50
mesh = generate_mesh(square,Nx)

degree_poly=2
scalar_element = FiniteElement("CG", mesh.ufl_cell(), degree_poly)
vector_element = VectorElement("CG", mesh.ufl_cell(), degree_poly)
system_element = MixedElement( vector_element , scalar_element )
dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
W = FunctionSpace(mesh,system_element)


# Define trial and test functions

up = Function(W)
(u, p) = split(up)

vq = TestFunction(W)
(v, q)  = split(vq)


# Define boundary conditions
noslip  = DirichletBC(W.sub(0), (0, 0),
                      "on_boundary && \
                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                        x[0] > 1.0 - DOLFIN_EPS)")
inflow  = DirichletBC(W.sub(0), (u_top, 0), "x[1] > 1.0 - DOLFIN_EPS")
#outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")

class CenterDomain(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.5, DOLFIN_EPS) and near(x[1], 0.5, DOLFIN_EPS)
center_domain = CenterDomain()

g2 = Constant(0.)
bc_one_point = DirichletBC(W.sub(1), g2, center_domain, method='pointwise')


bcs = [noslip, inflow, bc_one_point]


# Define the forms
h = function.specialfunctions.CellDiameter(mesh)
hmin = mesh.hmin()

def sigma_star(v):
    return v

c1 = Constant(1.)
c2 = Constant(1.)
nu_local = nu

b_form = 0.5*(inner(dot(u,grad(u)),v)  + inner(dot(u,grad(v)),u) )*dx
a_form = 2*nu*inner(sym(grad(u)),sym(grad(v)))*dx

tau_den = c1*(nu+nu_local)/(h/degree_poly)**2+c2*project(sqrt(u[0]**2+u[1]**2),V0)/(h/degree_poly)
tau    = project(1./tau_den,V0)
s_conv = (tau*inner(sigma_star(dot(u,grad(u))),sigma_star(dot(u,grad(v))))) *dx
s_pres = (tau*inner(sigma_star(grad(p)),sigma_star(grad(q)))) *dx

F = b_form + a_form - inner(p,div(v))*dx+ s_conv + inner(div(u),q)*dx + s_pres

J = derivative(F, up)


# Prepare nonlinear solver
snes_solver_parameters = {"nonlinear_solver":"snes",
                          "snes_solver": {"method" : "ngmres",
                                          "linear_solver": "mumps",
                                          "maximum_residual_evaluations":10000,
                                          "maximum_iterations": 1000,
                                          "report": True,
                                          "error_on_nonconvergence": False}
                          }

# Rey = 3000
# newtonls not converging (slow)
# basic not converging (slow)
# nrichardson diverges
# qn oscillatory , convergence really slow and oscillatory
# ncg diverges (fast)
# ngmres slowly slowly converging? I stopped at 200 iter 1.27e-02, 500 iter 4.2e-03, 1000 iter 2.1e-03 
# fas segmentation fault
# nasm error
# aspin error
# ngs error
# anderson fast but sloooowly converging. 200 iter 1.87e-02, iter 500 err 4e-03, iter 1000 2e-03, 1500 1.9e-03, 2000 1.5e-03
# ms too slow, not seeing anything
# composite segmentation fault error

problem = NonlinearVariationalProblem(F, up, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)

# Export the initial solution (zero)
outfile_u = File("lid-driven_cavity/u.pvd")
outfile_p = File("lid-driven_cavity/p.pvd")
outfile_ld = File("lid-driven_cavity/ld.pvd")

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
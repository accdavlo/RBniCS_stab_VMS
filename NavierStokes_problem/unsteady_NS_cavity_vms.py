from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

import matplotlib.pyplot as plt

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

with_plot=True
# Set parameter values
Re= Constant(10000. )
nu = Constant(1./Re)
f = Constant((0., 0.))
u_top = Constant(1.)

# Create mesh
square = Rectangle(Point(0., 0.), Point(1., 1.))

Nx=50
mesh = generate_mesh(square,Nx)

# XXX Time discretization
dt = 0.5
T = 1000.


degree_poly=2
scalar_element = FiniteElement("CG", mesh.ufl_cell(), degree_poly-1)
vector_element = VectorElement("CG", mesh.ufl_cell(), degree_poly)
system_element = MixedElement( vector_element , scalar_element )
dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
W = FunctionSpace(mesh,system_element)


# Define trial and test functions

up = Function(W)
(u, p) = split(up)

up_diff = Function(W)

vq = TestFunction(W)
(v, q)  = split(vq)

delta_up           = TrialFunction(W) # Trial function in the mixed space (Note: for the increment!)
(delta_u, delta_p) = split(delta_up) # Function in each subspace to write the functional  (Note: for the increment!)

# XXX Solution at the previous time
up_prev = Function(W)
(u_prev, _) = split(up_prev)




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


K = JacobianInverse(mesh)
G = K.T*K
gg = (K[0,0] + K[1,0])**2 + (K[0,1] + K[1,1])**2

rm = grad(p) + grad(u)*u - nu*(u.dx(0).dx(0) + u.dx(1).dx(1)) - f
rc = div(u)


tm=(4*((dt)**(-2)) +36*(nu**2)*inner(G,G) + inner(u,G*u))**(-0.5)
tc=(tm*gg)**(-1)

tcross = outer((tm*rm),(tm*rm))

F = (   inner(grad(v)*u+grad(q),tm*rm)*dx
        +inner(div(v),tc*rc)*dx
        +inner(grad(v).T*u,tm*rm)*dx
        -inner(grad(v),tcross)*dx
        + inner(v,(u-u_prev)/dt)*dx
      - inner(grad(v),outer(u,u))*dx 
      - inner(div(v),p)*dx + inner(q,div(u))*dx 
      + inner(sym(grad(v)),2*nu*sym(grad(u)))*dx  
       -inner(f,v)*dx  )
J = derivative(F, up, delta_up)

# Prepare nonlinear solver
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

problem = NonlinearVariationalProblem(F, up, bcs, J)
solver  = NonlinearVariationalSolver(problem)
#solver.parameters.update(snes_solver_parameters)

# Export the initial solution (zero)
outfile_u = File("lid-driven_cavity_unsteady/u.pvd")
outfile_p = File("lid-driven_cavity_unsteady/p.pvd")
outfile_ld = File("lid-driven_cavity_unsteady/ld.pvd")

solver.solve()
# Plot
(u, p) = up.split()
outfile_u << u
outfile_p << p


sav_ts = float (5) 

# XXX Time loop
K = int(T/dt)
for i in range(1,K):
    # Compute the current time
    t = i*dt
    print("t =", t)
    # Solve the nonlinear problem
    solver.solve()
    # Store the solution in up_prev
    up_diff.assign(up - up_prev)
    res = up_diff.vector().norm('l2')

    assign(up_prev, up)

    print(f"Residual {res}")
    # Plot
    (u, p) = up.split()
    if (i/sav_ts).is_integer():
        outfile_u << u
        outfile_p << p
    


plt.figure(1)
pp=plot(p); plt.colorbar(pp)
plt.title("Pressure")
plt.show(block=False)

plt.figure(2)
pp=plot(u[0]); plt.colorbar(pp)
plt.title("u")
plt.show(block=False)

plt.figure(3)
pp=plot(u[1]); plt.colorbar(pp)
plt.title("v")
plt.show(block=False)


plt.figure()
pp=plot(u); plt.colorbar(pp)
plt.title("Velocity")
plt.show()
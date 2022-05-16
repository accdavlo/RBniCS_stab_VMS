from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

import matplotlib.pyplot as plt

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

linear_implicit = True

with_plot = True
# Set parameter values
Re= Constant(1000. )
nu = Constant(1./Re)
f = Constant((0., 0.))
u_top = Constant(1.)

# Create mesh
square = Rectangle(Point(0., 0.), Point(1., 1.))

Nx=100
mesh = generate_mesh(square,Nx)

# XXX Time discretization
dt = 0.01
T = 10.

degree_poly=2
scalar_element = FiniteElement("CG", mesh.ufl_cell(), degree_poly)
vector_element = VectorElement("CG", mesh.ufl_cell(), degree_poly)
system_element = MixedElement( vector_element , scalar_element )
dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
W = FunctionSpace(mesh,system_element)


# Define trial and test functions

if linear_implicit:
    up = TrialFunction(W)
    (u, p) = split(up)
else:
    up = Function(W)
    (u, p) = split(up)


up_prev = Function(W)
(u_prev, p_prev) = split(up_prev)


up_pred = Function(W)
(u_pred, p_pred) = split(up_pred)

up_diff = Function(W)

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

tau_den = c1*(nu)/(h/degree_poly)**2+c2*project(sqrt(u_prev[0]**2+u_prev[1]**2),V0)/(h/degree_poly)
tau_v    = project(1./(1./dt+tau_den),V0)
tau_p = tau_v

tau_d = project((h/degree_poly)**2/c1*tau_den,V0)

if linear_implicit:
    b_form = 0.5*(0.5*(inner(dot(u_prev,grad(u_prev)),v)  + inner(dot(u_prev,grad(v)),u))\
                + 0.5*(inner(dot(u_pred,grad(u)),v)       + inner(dot(u_pred,grad(v)),u) ))*dx
    s_conv = 0.5*(tau_v*inner(sigma_star(dot(u_prev,grad(u_prev))),sigma_star(dot(u_prev,grad(u_prev))))\
                + tau_v*inner(sigma_star(dot(u_pred,grad(u))),sigma_star(dot(u_pred,grad(v)))))  *dx

else:
    b_form = 0.5*(0.5*(inner(dot(u_prev,grad(u_prev)),v)  + inner(dot(u_prev,grad(v)),u_prev)) \
                + 0.5*(inner(dot(u,grad(u)),v)  + inner(dot(u,grad(v)),u)) )*dx
    s_conv = 0.5*(tau_v*inner(sigma_star(dot(u_prev,grad(u_prev))),sigma_star(dot(u_prev,grad(v))))\
                + tau_v*inner(sigma_star(dot(u,grad(u))),sigma_star(dot(u,grad(v))))  ) *dx

a_form = nu*inner(sym(grad(u+u_prev)),sym(grad(v)))*dx


s_div  = 0.5*(tau_d*inner(sigma_star(div(u+u_prev)),sigma_star(div(v)))) *dx
s_pres = 0.5*(tau_p*inner(sigma_star(grad(p+p_prev)),sigma_star(grad(q)))) *dx

F = inner((u-u_prev)/dt,v)*dx + b_form + a_form - 0.5*inner(p+p_prev,div(v))*dx+ s_div\
    +s_conv + 0.5*inner(div(u+u_prev),q)*dx + s_pres


if linear_implicit:
    ll = lhs(F)
    rr = rhs(F)

    up = Function(W)
    (u, p) = split(up)
else:
    J = derivative(F, up)
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

(u,p) = up.split()
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
    if linear_implicit:
        for irk in range(2):
            assign(up_pred, up)
            LL = assemble(ll)
            RR = assemble(rr)
            [bc.apply(LL,RR) for bc in bcs]
            solve(LL,up.vector(),RR)
    else:
        for irk in range(2):
            assign(up_pred,up)
            solver.solve()
    # Store the solution in up_prev
    up_diff.assign(up - up_prev)
    diff_norm = up_diff.vector().norm('l2')
    res = norm(assemble(F))

    assign(up_prev, up)

    print(f"Residual {res}, step norm {diff_norm}")
    # Plot
    (u, p) = up.split()
    if (i/sav_ts).is_integer():
        outfile_u << u
        outfile_p << p
    



# solver.solve()
# # Plot
# (u, p) = up.split()
# outfile_u << u
# outfile_p << p

plt.figure()
pp=plot(p); plt.colorbar(pp)
plt.title("Pressure")
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

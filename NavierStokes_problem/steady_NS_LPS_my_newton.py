from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

from problems import Problem
import matplotlib.pyplot as plt
import numpy as np
import petsc4py
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;


# Newton tolerance on increment of u and p
tol=1e-3
max_iter = 30


# Create mesh
Nx=30
problem_name = "lid-driven_cavity"#"cylinder"#"lid-driven_cavity"
physical_problem = Problem(problem_name, Nx)
mesh = physical_problem.mesh
space_dim = physical_problem.space_dim

# Set parameter values
Re= Constant(1000. )
nu = Constant(1./Re)
f = Constant((0., 0.))
u_top = Constant(1.)

# Definition of all spaces we will need: CG2, CG1 (denoted as _proj), DG1 (denoted as _proj_DG)
degree_poly=2
scalar_element = FiniteElement("CG", mesh.ufl_cell(), degree_poly)
vector_element = VectorElement("CG", mesh.ufl_cell(), degree_poly)
scalar_element_proj = FiniteElement("CG", mesh.ufl_cell(), degree_poly-1)
vector_element_proj = VectorElement("CG", mesh.ufl_cell(), degree_poly-1)
scalar_element_proj_DG = FiniteElement("DG", mesh.ufl_cell(), degree_poly-1)
vector_element_proj_DG = VectorElement("DG", mesh.ufl_cell(), degree_poly-1)
system_element = MixedElement( vector_element , scalar_element )
system_element_proj = MixedElement( vector_element_proj , scalar_element_proj )
system_element_proj_DG = MixedElement( vector_element_proj_DG , scalar_element_proj_DG )
dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
W = FunctionSpace(mesh,system_element)
W_proj = FunctionSpace(mesh,system_element_proj)
W_proj_DG = FunctionSpace(mesh,system_element_proj_DG)
V = FunctionSpace(mesh,vector_element)
V_proj = FunctionSpace(mesh,vector_element_proj)



# Utils functions

def eliminate_zeros(sparse_M):
    # Input: scipy sparse matrix, remove all data smaller than 3e-14 in abs value
    sparse_M.data[np.abs(sparse_M.data)<3e-14] = 0.
    sparse_M.eliminate_zeros()
    return sparse_M

def PETSc2scipy(M):
    # Takes in input dolfin.cpp.la.Matrix and returns the csr_matrix in scipy
    s1 = M.size(0)
    s2 = M.size(1)
    (ai,aj,av) = as_backend_type(M).mat().getValuesCSR()
    sparse_M = sparse.csr_matrix((av,aj,ai), shape= [s1,s2] )
    sparse_M = eliminate_zeros(sparse_M)
    return sparse_M

def scipy2PETSc(M):
    # Input: scipy sparse matrix, returns a dolfin.cpp.la.PETScMatrix (used for system solutions)
    M=sparse.csr_matrix(M)
    aj = M.indices
    ai = M.indptr
    av = M.data
    PP = petsc4py.PETSc.Mat().createAIJ(size = M.shape, csr=(ai,aj,av))
    PP = dolfin.cpp.la.PETScMatrix(PP)
    return PP

def invert_block_matrix(S_diag):
    # Input a block_diagonal matrix (ex. DG), returns its inverse
    S_diag_inv = S_diag.copy()
    for i in range(len(S_diag.data)):
        if np.sum(np.abs(S_diag.data[i]))!=0:
            S_diag_inv.data[i] = np.linalg.inv(S_diag.data[i])
    return S_diag_inv

def lumping(M):
    # Input a scipy sparse matrix, returns a diagonal matrix with element the 
    # sum of the rows of the input
    data_lumping = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        data_lumping[i] = sum(M[i,:].data)
    return sparse.dia_matrix((data_lumping,0), shape=M.shape)

def inverse_lumping(M):
    # Input a scipy sparse matrix, returns a diagonal matrix with element the 
    # inverse of the sum of the rows of the input
    data_lumping = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        data_lumping[i] = 1./sum(M[i,:].data)
    return sparse.dia_matrix((data_lumping,0), shape=M.shape)

def brute_lumping_rectangular(M):
    # Chooses only one element per row and keeps it with the average value of 
    # the row. Used to pass from interpolation CG1->DG1 to projection DG1->CG1
    M = sparse.csr_matrix(M)
    L = sparse.lil_matrix(M.shape)
    for irow in range(M.shape[0]):
        row_start = M.indptr[irow]
        row_end = M.indptr[irow+1]
        column_first = M.indices[row_start]
        columns = M.indices[row_start:row_end]
        values = M.data[row_start:row_end]
        L[irow, column_first] = np.mean(values)

    return sparse.csr_matrix(L)


# Define the physiscal problem

bcs = physical_problem.define_bc(W, u_top)


# Define trial and test functions

up = Function(W)
(u, p) = split(up)
uptrial = TrialFunction(W)
(utrial, ptrial) = split(uptrial)
delta_up           = TrialFunction(W) # Trial function in the mixed space (XXX Note: for the increment!)
(delta_u, delta_p) = split(delta_up) # Function in each subspace to write the functional  (XXX Note: for the increment!)
vq = TestFunction(W)
(v, q)  = split(vq)
updg1 = TrialFunction(W_proj_DG)
vqdg1 = TestFunction(W_proj_DG)
(udg1, pdg1 ) =split(updg1)
(vdg1, qdg1 ) =split(vqdg1)
upcg1 = TrialFunction(W_proj)
vqcg1 = TestFunction(W_proj)


# Interpolation and projection between CG1 and CG2 and filter
x_u = inner(grad(utrial), grad(v))*dx
x_p = ptrial*q*dx
X_u = assemble(x_u)
X_p = assemble(x_p)

scalar_prod_form = inner(uptrial,vq)*dx
scalar_prod_matrix = assemble(scalar_prod_form)


# PETScDMCollection.create_transfer_matrix(coarse space, fine space) interpolation matrix from coarse to fine space
inter_CGpm1_2_CGp = PETScDMCollection.create_transfer_matrix(W_proj,W)
inter_CGpm1_2_CGp =inter_CGpm1_2_CGp.mat() #petsc4py.PETSc.Mat
proj_CGp_2_CGpm1 = PETScDMCollection.create_transfer_matrix(W,W_proj)
proj_CGp_2_CGpm1 = proj_CGp_2_CGpm1.mat()                   #petsc4py.PETSc.Mat

projection_MatrixCG = inter_CGpm1_2_CGp.matMult(proj_CGp_2_CGpm1)  #petsc4py.PETSc.Mat
projection_MatrixCG = PETScMatrix(projection_MatrixCG)  #dolfin.cpp.la.PETScMatrix
projection_MatrixCG_scipy = eliminate_zeros(PETSc2scipy(projection_MatrixCG))  #scipy.sparse
filterMatrixCG_scipy = eliminate_zeros(sparse.eye(projection_MatrixCG_scipy.shape[0]) -projection_MatrixCG_scipy )
polation_MatrixDG))  #scipy.sparse


# Grad operators from CG2 to DG1

inner_grad = inner(udg1,grad(q))*dx   # u * S* D* q  -> S*D

inner_dx=[]
for dim in range(space_dim):
    inner_dx.append( inner(uptrial.dx(dim), vqdg1)*dx )

scalar_dg1 = inner(udg1,vdg1)*dx + inner(pdg1,qdg1)*dx  # u * S *q -> S
X_inner_grad = PETSc2scipy(assemble(inner_grad))

X_inner_dx =[]
for dim in range(space_dim):
    X_inner_dx.append( PETSc2scipy(assemble(inner_dx[dim])) )

X_scalar_dg1 = PETSc2scipy(assemble(scalar_dg1))

S_diag_DG1 = sparse.bsr_matrix(X_scalar_dg1)
S_diag_DG1_inv = invert_block_matrix(S_diag_DG1) 

Deriv_cg2_to_dg1 =  [] 
for dim in range(space_dim):
    Deriv_cg2_to_dg1.append( sparse.csr_matrix(S_diag_DG1_inv@X_inner_dx[dim]) )


# Projections and interpolations between DG1 and CG1
scalar_cg1  = inner(upcg1,vqcg1)*dx
X_scalar_cg1 = PETSc2scipy(assemble(scalar_cg1))

scalar_cg_dg = inner(upcg1,vqdg1)*dx
scalar_dg_cg = inner(updg1,vqcg1)*dx

X_scalar_cg_dg = PETSc2scipy(assemble(scalar_cg_dg))
X_scalar_dg_cg = PETSc2scipy(assemble(scalar_dg_cg))

interpolation_cg1_to_dg1 = eliminate_zeros(sparse.csr_matrix(S_diag_DG1_inv@X_scalar_cg_dg))

#Projection DG1->CG1
# #Option 1 (+ dense)
# projection_dg1_to_cg1 = sparse.csr_matrix(interpolation_cg1_to_dg1.copy().transpose())
# projection_dg1_to_cg1 = normalize(projection_dg1_to_cg1,norm='l1', axis=1)

# # #Option 2 (less dense) I don't know how to implement it
# projection_dg1_to_cg1 = brute_lumping_rectangular(interpolation_cg1_to_dg1.copy().transpose())

# # #Option 3 Super long to be computed!! Filling the memory -> Find a better way to interpolate
# projection_dg1_to_cg1 = spsolve(X_scalar_cg1, X_scalar_dg_cg)

#Option 4 Simplification of option 3 with mass lumping
X_scalar_cg1_lumped_inv = inverse_lumping(X_scalar_cg1)
projection_dg1_to_cg1 = sparse.csr_matrix(X_scalar_cg1_lumped_inv@X_scalar_dg_cg)


projection_MatrixDG_scipy = interpolation_cg1_to_dg1@projection_dg1_to_cg1
filterMatrixDG_scipy = eliminate_zeros(sparse.eye(projection_MatrixDG_scipy.shape[0]) - projection_MatrixDG_scipy)  #scipy.sparse

projection_MatrixDG = scipy2PETSc(projection_MatrixDG_scipy)
filterMatrixDG      = scipy2PETSc(filterMatrixDG_scipy)


# Define the forms
h = function.specialfunctions.CellDiameter(mesh)
hmin = mesh.hmin()
n = FacetNormal(mesh)

# Constants
# Stabilization coefficients: the smaller c1, c2, the larger the stabilization
c1 = Constant(0.1)
c2 = Constant(0.1)
# Smagorinskij constant
CS = Constant(0.1)
nu_local = nu

# Fully Nonlinear for automatic nonlinear solvers
b_form = 0.5*(inner(dot(u,nabla_grad(u)),v)  - inner(dot(u,nabla_grad(v)),u) )*dx +0.5*dot(dot(v,outer(u,u)),n)*ds
a_form = 2*nu*inner(sym(grad(u)),sym(grad(v)))*dx

tau_den = c1*(nu+nu_local)/(h/degree_poly)**2+ c2*project(sqrt(u[0]**2+u[1]**2),V0)/(h/degree_poly)
tau    = project(1./tau_den,V0)
s_conv = (tau*inner(dot(u,nabla_grad(u)),dot(u,nabla_grad(v)))) *dx
s_pres = (tau*inner(grad(p),grad(q))) *dx

F = b_form + a_form - inner(p,div(v))*dx+ s_conv + inner(div(u),q)*dx + s_pres

J = derivative(F, up, delta_up)


# Newton forms: Jacobian (LL) and residual (RR)
def assemble_forms(up):
    (u,p)=split(up)

    tau_den = c1*(nu+nu_local)/(h/degree_poly)**2+c2*project(sqrt(u[0]**2+u[1]**2),V0)/(h/degree_poly)
    tau    = project(1./tau_den,V0)
    
    # For Jacobian
    b_form_lin = 0.5*(inner(dot(delta_u,nabla_grad(u)),v) + inner(dot(u,nabla_grad(delta_u)),v)  \
                - inner(dot(delta_u,nabla_grad(v)),u)- inner(dot(u,nabla_grad(v)),delta_u) )*dx \
                +0.5*dot(dot(v,outer(u,delta_u)),n)*ds +0.5*dot(dot(v,outer(delta_u,u)),n)*ds
    a_form_lin = 2*nu*inner(sym(grad(delta_u)),sym(grad(v)))*dx
    s_conv_lin = (tau*inner(dot(u,nabla_grad(delta_u)),dot(u,nabla_grad(v)))) *dx
                 # (tau*inner(dot(delta_u,nabla_grad(u)),dot(u,nabla_grad(v)))) *dx+\
                 # (tau*inner(dot(u,nabla_grad(delta_u)),dot(u,nabla_grad(v)))) *dx+\
                 # (tau*inner(dot(u,nabla_grad(u)),dot(delta_u,nabla_grad(v)))) *dx
    s_pres_lin = (tau*inner(grad(delta_p),grad(q))) *dx

    # For residual
    b_form = 0.5*(inner(dot(u,nabla_grad(u)),v)  - inner(dot(u,nabla_grad(v)),u) )*dx +0.5*dot(dot(v,outer(u,u)),n)*ds
    a_form = 2*nu*inner(sym(grad(u)),sym(grad(v)))*dx
    s_conv = (tau*inner(dot(u,nabla_grad(u)),dot(u,nabla_grad(v)))) *dx
    s_pres = (tau*inner(grad(p),grad(q))) *dx
    # S_pres = assemble(s_pres)


    # Smagorinskij term
    tau_sma = (CS * h )**2 *sqrt(inner(0.5*(grad(u) + nabla_grad(u) ) , 0.5*(grad(u) + nabla_grad(u) ) ) )
    base_matrix_sma =  PETSc2scipy(assemble(tau_sma* inner(pdg1,qdg1)*dx))

    sma_matrix = []
    for dim in range(space_dim):
        DF = Deriv_cg2_to_dg1[dim] @ filterMatrixCG_scipy
        sma_matrix.append( eliminate_zeros( \
            DF.copy().transpose() @ base_matrix_sma @ DF ) \
        )
    sma_matrix = scipy2PETSc(sum(sma_matrix))

    # # Stabilizations

    # # Velocity stabilization term
    # # tau*(sigma^*( u grad(u)), sigma^*(u grad v) )*dx )
    base_matrix_conv_stab = PETSc2scipy(assemble(tau* inner(udg1,vdg1)*dx))
    
    Convection_cg2_to_dg1 = []
    for dim in range(space_dim):
        Convection_cg2_to_dg1.append(\
            sparse.csr_matrix(S_diag_DG1_inv@ PETSc2scipy( assemble( inner(delta_u.dx(dim)*u[dim], vdg1 )*dx ) ) )\
        )

    conv_stab_matrix=[]
    for dim in range(space_dim):
        # FC = filterMatrixDG_scipy@ Convection_cg2_to_dg1[dim]
        FC = Convection_cg2_to_dg1[dim]
        conv_stab_matrix.append(\
            eliminate_zeros( \
                FC.transpose() @ base_matrix_conv_stab @ FC \
            )\
        )
    conv_stab_matrix = sum(conv_stab_matrix)



    # Convection_cg2_to_dg1 = []
    # for component in range(space_dim):
    #     Convection_cg2_to_dg1.append(\
    #         sparse.csr_matrix(S_diag_DG1_inv@ PETSc2scipy( assemble( inner(dot(u,nabla_grad(delta_u[component])), vdg1[component] )*dx ) ) )\
    #     )

    # conv_stab_matrix=[]
    # for component in range(space_dim):
    #     # FC = filterMatrixDG_scipy@ Convection_cg2_to_dg1[dim]
    #     FC = Convection_cg2_to_dg1[component]
    #     conv_stab_matrix.append(\
    #         eliminate_zeros( \
    #             FC.transpose() @ base_matrix_conv_stab @ FC \
    #         )\
    #     )
    # conv_stab_matrix = sum(conv_stab_matrix)



    # Convection_cg2_to_dg1 = sparse.csr_matrix(\
    #         S_diag_DG1_inv@ PETSc2scipy( assemble( \
    #         inner(dot(u,nabla_grad(delta_u)), vdg1 )*dx ) ) )

    # # FC = filterMatrixDG_scipy@ Convection_cg2_to_dg1[dim]
    # FC = Convection_cg2_to_dg1
    # conv_stab_matrix = eliminate_zeros( \
    #         FC.transpose() @ base_matrix_conv_stab @ FC )



    conv_stab_matrix = scipy2PETSc(conv_stab_matrix) # dolfin.cpp.la.PETScMatrix

    # s_conv_lin = (tau*inner(sigma_star(dot(delta_u,nabla_grad(u))),sigma_star(dot(u,nabla_grad(v))))) *dx+\
    #              (tau*inner(sigma_star(dot(u,nabla_grad(delta_u))),sigma_star(dot(u,nabla_grad(v))))) *dx+\
    #              (tau*inner(sigma_star(dot(u,nabla_grad(u))),sigma_star(dot(delta_u,nabla_grad(v))))) *dx



    # # Pression stabilization term 
    base_matrix_pres_stab = PETSc2scipy(assemble(tau* inner(pdg1,qdg1)*dx))

    pres_stab_matrix = []
    for dim in range(space_dim):
        # FD = Deriv_cg2_to_dg1[dim]
        FD = filterMatrixDG_scipy @ Deriv_cg2_to_dg1[dim]
        pres_stab_matrix.append(\
            eliminate_zeros( \
                FD.copy().transpose() @ base_matrix_pres_stab @ FD ) \
        )

    pres_stab_matrix = sum(pres_stab_matrix)
    pres_stab_matrix = scipy2PETSc(pres_stab_matrix) # dolfin.cpp.la.PETScMatrix



    lhs_form = b_form_lin + a_form_lin - inner(delta_p,div(v))*dx + inner(div(delta_u),q  )*dx  + s_conv_lin #   + s_pres_lin
    rhs_form = -(b_form + a_form - inner(p,div(v))*dx + inner(div(u),q)*dx   + s_conv )# ) + s_pres 

    LL = as_backend_type(assemble(lhs_form))+ sma_matrix +pres_stab_matrix #+ conv_stab_matrix
    RR = as_backend_type(assemble(rhs_form))

    rhs_pres_stab = RR.copy() 
    pres_stab_matrix.mult(up.vector(), rhs_pres_stab)

    rhs_conv_stab = RR.copy() 
    conv_stab_matrix.mult(up.vector(), rhs_conv_stab)

    RR = RR - rhs_pres_stab # -rhs_conv_stab
    
    return LL, RR

# # Prepare nonlinear solver

# snes_solver_parameters = {"nonlinear_solver":"snes",
#                           "snes_solver": {"method" : "anderson",
#                                           "linear_solver": "mumps",
#                                           "maximum_residual_evaluations":10000,
#                                           "maximum_iterations": 1000,
#                                           "report": True,
#                                           "error_on_nonconvergence": False}
#                           }

# # snes_solver_parameters = {"nonlinear_solver": "snes",
# #                           "snes_solver": {"linear_solver": "mumps",
# #                                           "maximum_iterations": 20,
# #                                           "report": True,
# #                                           "error_on_nonconvergence": True}}

# # Rey = 10000
# # Newton by FEniCS not converging
# # SNES solvers:
# # newtonls not converging (super slow)
# # nrichardson diverges
# # qn diverges
# # ncg diverges (fast)
# # ngmres slowly slowly converging? I stopped at 1000 iter 1.e-04 
# # fas segmentation fault
# # nasm error
# # aspin error
# # ngs error
# # anderson fast but sloooowly converging. 1000 iter 1.e-04
# # ms too slow, not seeing anything
# # composite segmentation fault error

bcs[0].apply(up.vector())

dup = Function(W)

norm_increment_u = 2*tol
norm_increment_p = 2*tol

bcdu_inflow =  DirichletBC(W.sub(0), (0., 0), "x[1] > 1.0 - DOLFIN_EPS")

bcs_dup = [bcdu_inflow]
[bcs_dup.append(bc) for bc in bcs[1:]]

for it in range(max_iter):

    LL, RR =  assemble_forms(up)
    [ bc.apply(LL,RR) for bc in bcs_dup]
    solve(LL,dup.vector(),RR )
    #solve(lhs_form == rhs_form, dup, bcs_dup)
    # F_vec = assemble(F)
    # [bc.apply(F_vec) for bc in bcs]
    # J_mat = assemble(J)
    # [bc.apply(J_mat) for bc in bcs]
    # solve(J_mat, dup.vector(), F_vec)
    up.vector().add_local(dup.vector().get_local())
    up.vector().apply("")
    #bcs[0].apply(up.vector())

    err_relative = norm(dup)
#    err_abs = norm(F_vec)
    print(f"Relative tolerance {err_relative}")#, absolute_tolerance {err_abs}")
    # Compute the (relative) norm of the increment
    # Notice that the solution update and norm computation
    # steps have been swapped when compared to 2(a).
    norm_increment_u = (
        dup.vector().inner(X_u*dup.vector()) /
              up.vector().inner(X_u*      up.vector())
    )
    norm_increment_p = (
        dup.vector().inner(X_p*dup.vector()) /
              up.vector().inner(X_p*      up.vector())
    )
    # Print to screen
    print("Iteration", it,
        "|| delta_u ||_V / || u ||_V =", norm_increment_u,
        "|| delta_p ||_Q / || p ||_Q =", norm_increment_p)
    # Check stopping criterion
    if (norm_increment_u < tol and norm_increment_p < tol):
        print("Fixed point loop has converged after", it, "iterations.")
        break



# Export the initial solution (zero)
outfile_u = File(physical_problem.name+"_steady/u.pvd")
outfile_p = File(physical_problem.name+"_steady/p.pvd")
outfile_ld = File(physical_problem.name+"_steady/ld.pvd")

#solver.solve()



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

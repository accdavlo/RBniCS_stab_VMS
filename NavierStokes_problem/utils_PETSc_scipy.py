from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

import numpy as np
import petsc4py
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize

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


def inverse_lumping_with_zeros(M):
    # Input a scipy sparse matrix, returns a diagonal matrix with element the 
    # inverse of the sum of the rows of the input
    data_lumping = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        ss=sum(M[i,:].data)
        if abs(ss)>1e-14:
            data_lumping[i] = ss
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

'''
TOOLS FOR COMPRESSIVE SENSING AND SPARSE SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH, INDIAN INSTITUTE OF SCIENCE, BANGALORE
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% IMPORT LIBRARIES
import numpy as np
import utils

# %% GREEDY ALGORITHMS

def basic_thresholding():
    return

def matching_pursuit():
    return

def orthogonal_matching_pursuit(y, A, max_iter=100, tol=1e-6):
    '''
    Sparse recovery using Orthogonal Matching Pursuit (OMP)
    
    :param y: Measurement vector
    :param A: Sensing matrix
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance of error

    :returns: Sparse vector that solves y = Ax
    :returns: Error at each iteration

    '''

    y = y[:, np.newaxis]
    _, n = A.shape
    x = np.zeros((n,1), dtype=complex)
    supp = []
    error = []
    residual = y

    for _ in range(max_iter):
        projection = np.conj(A.T).dot(residual)
        idx = np.argmax(np.abs(projection))

        supp = np.union1d(supp, [idx])
        supp = np.sort(supp)

        As = A[:, supp.astype(int)]
        xs = np.linalg.pinv(As).dot(y)
        x[supp.astype(int)] = xs

        residual = y - A.dot(x)
        error_iter = np.linalg.norm(residual)**2
        error.append(error_iter)
        if error_iter < tol:
            break

    return np.squeeze(np.real(x), axis=1), error

def cosamp():
    return

def iterative_hard_thresholding():
    return

def subspace_pursuit():
    return

# %% BASIS PURSUIT

def iterative_soft_thresholding(y, A, lambd, max_iter=100, tol=1e-6):
    '''
    Sparse recovery using iterative soft-thresholding (ISTA)

    :param y: Measurement vector
    :param A: Sensing matrix
    :param lambd: Penalty parameter
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance of error

    :returns: Sparse vector that solves y = Ax
    :returns: Error at each iteration

    '''

    _, n = A.shape
    x = np.zeros(n,dtype=complex)

    AHA = np.conj(A.T).dot(A)
    eigval, _ = np.linalg.eig(AHA)
    t = 1.0/np.max(eigval)

    for _ in range(max_iter):

        xold = x
        x = utils.soft_thresholding(x+t*np.conj(A.T).dot(y-A.dot(x)),lambd*t)

        if (np.linalg.norm(x-xold)/np.linalg.norm(xold)<tol):
            break

    return x

def basis_pursuit_admm(y, A, lambd, rho=0.8, max_iter=100, tol=1e-6):
    
    # y = y[:, np.newaxis]
    # ## Initialization
    # m,n = np.shape(A)
    # x = np.zeros((n,1),dtype=complex)
    # u = np.zeros((n,1),dtype=complex)
    # z = np.zeros((n,1),dtype=complex)
    # loss = np.zeros(max_iter)

    # ## Static objects
    # AAt = A.dot(np.conj(A.T))
    # P = np.identity(n) - np.conj(A.T).dot(np.linalg.solve(AAt,A))
    # q = np.conj(A.T).dot(np.linalg.solve(AAt,y))

    # ## ADMM Iterations
    # for i in range(max_iter):

    #     # x update
    #     x = P.dot(z-u)[:,0] + q[:,0]
    #     x = x[:,np.newaxis]

    #     # z update
    #     zold = z
    #     xhat = lambd*x + (1-lambd)*zold
    #     z = utils.soft_thresholding(x+u,1.0/rho)

    #     # u update
    #     u = u + (xhat-z)

    #     # Stopping criterion
    #     loss[i] = (0.5)*np.linalg.norm(A.dot(z)-y)**2# + L*np.linalg.norm(z,1)
    #     if (np.linalg.norm(z-zold)/np.linalg.norm(zold)<tol):
    #         break

    # return np.squeeze(np.real(z), axis=1), loss
    return

def dantzig_selector():
    return

# %% ITERATIVE REWEIGHTED METHODS

def focuss():
    return
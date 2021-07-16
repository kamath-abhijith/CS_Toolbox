'''

ALGORITHMS FOR COMPRESSIVE SENSING AND SPARSE SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% IMPORT LIBRARIES

import numpy as np

# %% ACTIVATION FUNCTIONS

def shrinkage(x, lambd):
    ''' Soft-thresholding of x with threshold lambd '''
    return np.maximum(0,x-lambd) - np.maximum(0,-x-lambd)

# %% SPARSE RECOVERY ALGORITHMS

def l0_ihta():
    pass

def l1_bp():
    pass

def l1_ista(y, A, lambd, max_iter=100, tol=1e-12):
    '''
    Sparse recovery using iterative soft-thresholding (ISTA)

    :param y: Measurement vector
    :param A: Sensing matrix
    :param lambd: Penalty parameter
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance of error

    :returns: Sparse vector that solves y = Ax

    '''

    _, n = A.shape
    x = np.zeros((n,1),dtype=complex)

    AHA = np.conj(A.T).dot(A)
    eigval, _ = np.linalg.eig(AHA)
    t = 1.0/np.max(eigval)

    errors = []
    for _ in range(max_iter):

        xold = x
        x = shrinkage(x+t*np.conj(A.T)@(y-A@x),lambd*t)

        error = np.linalg.norm(y-A@x)**2
        errors.append(error)
        if (np.linalg.norm(x-xold)/np.linalg.norm(xold)<tol):
            break

    return np.real(x), errors

def l1_fista():
    pass

def l1_admm():
    pass
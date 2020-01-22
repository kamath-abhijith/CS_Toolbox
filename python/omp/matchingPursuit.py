import numpy as np

def greedy_mp(A,b,*argv):
    '''
    Sparse recovery using matching pursuit

    INPUT:  Measurement matrix, A
            Measurements, b
            Optional: Number of iterations
                      Error tolerance
    OUTPUT: Estimated sparse vector

    Author: Abijith J. Kamath
    kamath-abhijith.github.io
    '''

    ## Read optional arguments
    nargin = len(argv)
    if (nargin >= 1):
        iter_lim = argv[0]
    else:
        iter_lim = len(b)

    if (nargin >= 2):
        tol = argv[1]
    else:
        tol = 1e-6

    ## Initializations
    m,n = np.shape(A)
    bmp = np.zeros((m,1))
    x = np.zeros((n,1))
    res_b = b[:,0]

    ## Iterations
    for i in range(iter_lim):

        # Projection step
        weights = A.transpose().dot(res_b)
        idx = np.argmax(np.abs(weights))

        # Update step
        x[idx] = x[idx] + weights[idx]
        update = weights[idx]*A[:,idx]
        bmp = bmp + update

        # Residue corrections
        res_b = res_b - update

        # Stopping criteria
        e_rec = np.linalg.norm(b-A.dot(x))**2
        if (e_rec < tol):
            break

    return x

def greedy_omp(A,b,*argv):
    '''
    Sparse recovery using orthogonal matching pursuit

    INPUT:  Measurement matrix, A
            Measurements, b
            Optional: Number of iterations
                      Error tolerance
    OUTPUT: Estimated sparse vector

    Author: Abijith J. Kamath
    kamath-abhijith.github.io
    '''

    ## Read optional arguments
    nargin = len(argv)
    if (nargin >= 1):
        iter_lim = argv[0]
    else:
        iter_lim = len(b)

    if (nargin >= 2):
        tol = argv[1]
    else:
        tol = 1e-6


    ## Initialization
    m,n = np.shape(A)
    bomp = np.zeros((m,1))
    x = np.zeros((n,1))
    supp = []
    res_b = b[:,0]

    ## Iterations 
    for i in range(iter_lim):

        # Projection step
        proj = A.transpose().dot(res_b)
        idx = np.argmax(np.abs(proj))

        # Support update
        supp = np.union1d(supp,[idx])
        supp = np.sort(supp)

        # Orthogonal projection step
        As = A[:,supp.astype(int)]
        xs = np.linalg.pinv(As).dot(b)
        x[supp.astype(int)] = xs

        # Residue correction
        res_b = b[:,0] - A.dot(x)[:,0]

        # Stopping criteria
        e_rec = np.linalg.norm(b-A.dot(x))**2
        if (e_rec < tol):
            break

    return x
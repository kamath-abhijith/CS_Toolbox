import numpy as np

def shrinkage(x,a):
    '''
    INPUT:  Vector, x
            Thresholding parameter, a

    OUTPUT: Soft thresholded vector
    '''
    return np.maximum(0,x-a) - np.maximum(0,-x-a)

def greedy_mp(A,b,L,*argv):
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
    x = np.zeros((n,1),dtype=complex)
    res_b = b[:,0]

    ## Iterations
    for i in range(iter_lim):

        # Projection step
        weights = np.conj(A.transpose()).dot(res_b)
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
    x = np.zeros((n,1),dtype=complex)
    supp = []
    res_b = b[:,0]

    ## Iterations 
    for i in range(iter_lim):

        # Projection step
        proj = np.conj(A.transpose()).dot(res_b)
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

def pursuit_ista(A,b,L,*argv):
    '''
    Sparse recovery using Iterative
    Soft Thresholding (ISTA)

    INPUT:  Measurement matrix, A
            Measurements, b
            Sparsity penalty, L
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
    x = np.zeros((n,1),dtype=complex)
    
    ## Proximal weight
    AHA = np.conj(A.T).dot(A)
    eigval,eigvec = np.linalg.eig(AHA)
    t = 1.0/np.max(eigval)

    ## ISTA Iterations
    for i in range(int(iter_lim)):

        xold = x
        x = shrinkage(x+t*np.conj(A.T).dot(b-A.dot(x)),L*t)

        # Stopping criterion
        if (np.linalg.norm(x-xold)/np.linalg.norm(xold)<tol):
            break

    return x

def pursuit_admm(A,b,L,rho,*argv):
    '''
    Sparse recovery using basis pursuit
    Oprimization solved using ADMM

    INPUT:  Measurement matrix, A
            Measurements, b
            Relaxation weight, L
            Sparsity penalty, rho
            Optional: Number of iterations
                      Error tolerance

    OUTPUT: Estimated sparse vector

    Author: Abijith J. Kamath
    kamath-abhijith.github.io

    For more information, check: Lectures from Boyd
    https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf
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
    x = np.zeros((n,1),dtype=complex)
    u = np.zeros((n,1),dtype=complex)
    z = np.zeros((n,1),dtype=complex)
    loss = np.zeros(iter_lim)

    ## Static objects
    AAt = A.dot(np.conj(A.T))
    P = np.identity(n) - np.conj(A.T).dot(np.linalg.solve(AAt,A))
    q = np.conj(A.T).dot(np.linalg.solve(AAt,b))

    ## ADMM Iterations
    for i in range(iter_lim):

        # x update
        x = P.dot(z-u)[:,0] + q[:,0]
        x = x[:,np.newaxis]

        # z update
        zold = z
        xhat = L*x + (1-L)*zold
        z = shrinkage(x+u,1.0/rho)

        # u update
        u = u + (xhat-z)

        # Stopping criterion
        loss[i] = (0.5)*np.linalg.norm(A.dot(z)-b)**2# + L*np.linalg.norm(z,1)
        if (np.linalg.norm(z-zold)/np.linalg.norm(zold)<tol):
            break

    return z, loss

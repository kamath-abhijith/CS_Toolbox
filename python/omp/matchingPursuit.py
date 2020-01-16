import numpy as np

def mat_pursuit(A,b,iter_lim):
    m,n = np.shape(A)
    bmp = np.zeros((m,1))
    sparse_code = np.zeros((n,1))
    res_b = b[:,0]
    
    for i in range(iter_lim):
        weights = A.transpose().dot(res_b)
        idx = np.argmax(np.abs(weights))
        sparse_code[idx] = sparse_code[idx] + weights[idx]
        update = weights[idx]*A[:,idx]
        bmp = bmp + update
        res_b = res_b - update

    return sparse_code
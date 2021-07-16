'''
PHASE TRANSITION DIAGRAM OF SPARSE RECOVERY ALGORITHMS

AUTHOR; ABIJITH J. KAMATH, INDIAN INSTITUTE OF SCIENCE
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% IMPORT LIBRARIES
import numpy as np

from tqdm import tqdm

import utils
import cs_tools

# %% SET PARAMETERS
N_ITER = 100
THRESH = 1e-3
SNR = 60.0

N = 20
t = np.linspace(0,1,N)

success_map = np.zeros((N,N))

# %% START

for k in tqdm(range(N)):

    freq = 250.0*np.random.rand(k)                          # Frequencies
    amp = np.ones(k)                                        # Amplitudes
    theta = 2.0*np.pi*np.random.randn(k)                    # Phase differences

    signal = amp.dot(np.sin(np.outer(freq, t)))
    signal *= np.hanning(N)

    fourier_mtx = utils.fftmtx(N)
    inv_fourier_mtx = np.conj(fourier_mtx.T)
    true_sparse_vector = np.matmul(inv_fourier_mtx, signal)

    for m in range(N):
        
        sensing_mtx = np.random.randn(m,N)
        forward_mtx = np.matmul(sensing_mtx, inv_fourier_mtx)

        measurements = sensing_mtx.dot(signal)

        for iter in range(N_ITER):
            measurements += np.sqrt(np.linalg.norm(true_sparse_vector)*10**(-SNR/10.0))*np.random.randn(m)
            estimated_sparse_vector, _ = cs_tools.orthogonal_matching_pursuit(measurements,
                forward_mtx, max_iter=m, tol=1e-12)

            error_sparse_vector = utils.mean_squared_error(np.real(true_sparse_vector),
                np.real(estimated_sparse_vector))/np.linalg.norm(true_sparse_vector)

            if error_sparse_vector <= THRESH:
                success_map[k,m] += 1 

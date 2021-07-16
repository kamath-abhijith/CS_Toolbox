'''
PHASE TRANSITION DIAGRAM OF SPARSE RECOVERY ALGORITHMS

AUTHOR; ABIJITH J. KAMATH, INDIAN INSTITUTE OF SCIENCE
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% IMPORT LIBRARIES
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import style 
from matplotlib import rcParams

from joblib import Parallel, delayed
from scipy.io import savemat

import utils
import cs_tools

# %% PLOT SETTINGS
plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["cm"],  # specify font here
    "mathtext.fontset": "cm",
    "font.size": 26})

# path = '/Users/abhijith/Desktop/TECHNOLOGIE/Research/TimeEncodingMachines/Documentation/TEMFRI/TSP/figures/'

# %% SET PARAMETERS
METHOD = "OMP"
N_ITER = 100
THRESH = 1e-3
SNR = 60.0

N = 512
t = np.linspace(0,1,N)

N_STEP = 32
K_LIST = np.arange(1,N+1,N_STEP)
M_LIST = np.arange(1,N+1,N_STEP)

success_map = np.zeros((len(K_LIST),len(M_LIST)))

# %% SPARSE RECOVERY

def run_sparse_recovery(k_iter):
    k = K_LIST[k_iter]
    freq = 500.0*np.random.rand(k)                        # Frequencies
    amp = np.ones(k)                                      # Amplitudes
    # theta = 2.0*np.pi*np.random.randn(k)                    # Phase differences

    signal = amp.dot(np.sin(np.outer(freq, t)))
    signal *= np.hanning(N)

    fourier_mtx = utils.fftmtx(N)
    inv_fourier_mtx = np.conj(fourier_mtx.T)
    true_sparse_vector = np.matmul(inv_fourier_mtx, signal)

    for m_iter in tqdm(range(len(M_LIST))):
        m = M_LIST[m_iter]
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
                success_map[k_iter,m_iter] += 1

    return success_map

# %% START

if __name__ == '__main__':
    result = Parallel(n_jobs=8, backend="multiprocessing", verbose=10)(
        delayed(run_sparse_recovery)(k_iter) for k_iter in tqdm(range(len(K_LIST)))
        )

# %% PLOT HEAT MAP
plt.figure(figsize=(12,6))
ax = plt.gca()

success_map = sum(result)
normalised_map = success_map/np.max(success_map)
plot_success_map = np.flip(np.flip(normalised_map.T, 1), 0)

utils.plot_heatmap(plot_success_map, ax=ax, xaxis_label=r"$m/N$", yaxis_label=r"$k/N$",
    annotation=False)

# %% SAVE MAT FILE
savemat("N_"+str(N)+"_N_ITER_"+str(N_ITER)+"_STEP_"+str(N_STEP)+"_METHOD_"+METHOD+".mat",
    plot_success_map)

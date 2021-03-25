'''
SPARSE SIGNAL RECOVERY DEMO

AUTHOR: ABIJITH J. KAMATH, INDIAN INSTITUTE OF SCIENCE
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% IMPORT LIBRARIES
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

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

# %% CONSTRUCT SIGNAL
np.random.seed(43)

N = 512
t = np.linspace(0,1,N)

k = 1                                                   # Number of frequencies
freq = 500.0*np.random.rand(k)                          # Frequencies
amp = np.ones(k)                                        # Amplitudes
theta = 2.0*np.pi*np.random.randn(k)                    # Phase differences

signal = amp.dot(np.sin(np.outer(freq, t)))
signal *= np.hanning(N)

# %% PLOT ORIGINAL SIGNAL
utils.plot_signal(t, signal)

# %% SPARSE MEASUREMENTS
m = 40                                                  # Number of measurements

fourier_mtx = utils.fftmtx(N)
inv_fourier_mtx = np.conj(fourier_mtx.T)
true_sparse_vector = np.matmul(inv_fourier_mtx, signal)

sensing_mtx = np.random.randn(m,N)
forward_mtx = np.matmul(sensing_mtx, inv_fourier_mtx)

measurements = sensing_mtx.dot(signal)

# %% PLOT TRUE SPARSE VECTOR
utils.plot_sparse_vector(true_sparse_vector)

# %% SPARSE RECOVERY
estimated_sparse_vector, _ = cs_tools.orthogonal_matching_pursuit(measurements,
    forward_mtx, max_iter=m, tol=1e-12)
    
# %% PLOT ESTIMATED SPARSE VECTOR
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_sparse_vector(true_sparse_vector, ax=ax, line_width=4,
    plot_colour='green')
utils.plot_sparse_vector(estimated_sparse_vector, ax=ax, line_width=4,
    line_style='--', plot_colour='red', show=True)

# %% SIGNAL RECONSTRUCTON
reconstruction = np.real(fourier_mtx.dot(estimated_sparse_vector))

error_signal = utils.mean_squared_error(signal, reconstruction)
error_sparse_vector = utils.mean_squared_error(np.real(true_sparse_vector),
    np.real(estimated_sparse_vector))

print('MSE between signal: {:.2e}'.format(error_signal))
print('MSE between sparse vector: {:.2e}'.format(error_sparse_vector))

# %% PLOT SIGNAL RECOVERY
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(t, signal, ax=ax, plot_colour='green', line_width=4,
    legend_label=r"TRUE", legend_show=True)
utils.plot_signal(t, reconstruction, ax=ax, plot_colour='red', line_width=4,
    legend_label=r"RECOVERED", legend_show=True, show=True)
# %%

# %% IMPORT LIBRARIES

import numpy as np

from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

import cs_tools
import utils

# %% Load signal and dictionary
load_dict_f = loadmat('hw1problem3.mat')

A = load_dict_f['Psi']
b = load_dict_f['f']
b = np.squeeze(b, axis=1)

# %% SPARSE RECOVERY
MAX_ITER = 1000
TOL = 1e-12

method = 'admm'

if method == 'ista':
    lambd = 0.01
    x = cs_tools.iterative_soft_thresholding(b, A, lambd, max_iter=MAX_ITER, tol=TOL)

elif method == 'omp':
    x, _ = cs_tools.orthogonal_matching_pursuit(b, A, max_iter=MAX_ITER, tol=TOL)

elif method == 'admm':
    lambd = 0.01
    rho = 0.8
    x, _ = cs_tools.basis_pursuit_admm(b, A, lambd=lambd, rho=rho,
        max_iter=MAX_ITER, tol=TOL)

b_rec = A.dot(x)

error = b-b_rec
error = np.linalg.norm(error)
error = 20*np.log10(error)
print("SRNR: ",error,"dB")

# %% PLOT SPARSE
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_sparse_vector(x, ax=ax, show=True)

# %% PLOT MEASUREMENTS
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(np.arange(len(b)), b_rec, ax=ax, plot_colour='red')
utils.plot_signal(np.arange(len(b)), b, ax=ax, plot_colour='green', show=True)
# %%

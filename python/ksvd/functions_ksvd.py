## FUNCTIONS FOR K-SVD -- DENOISING AND IMAGE SUPER-RESOLUTION
#
# AUTHOR: ABIJITH J KAMATH, INDIAN INSTITUTE OF SCIENCE, BANGALORE
# FOR MORE INFORMATION, SEE: https://kamath-abhijith.github.io

import sys
import numpy as np
import timeit

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from scipy.sparse.linalg import svds
from skimage.util.shape import *
from skimage.util import pad
from operator import mul, sub
from functools import reduce
from tqdm import tqdm

# %% IMAGE TO PATCH -- PATCH TO IMAGE

def patch_matrix_windows(img, window_shape, step):
    patches = view_as_windows(img, window_shape, step=step)
    cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
    return cond_patches, patches.shape

def image_reconstruction_windows(mat_shape, patch_mat, patch_sizes, step):
    img_out = np.zeros(mat_shape)
    for l in range(patch_mat.shape[1]):
        i, j = divmod(l, patch_sizes[1])
        temp_patch = patch_mat[:, l].reshape((patch_sizes[2], patch_sizes[3]))
        img_out[i*step:(i+1)*step, j*step:(j+1)*step] = temp_patch[:step, :step].astype(int)
    return img_out

# %% SPARSE CODING
def omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    res = np.linalg.norm(vect_y)
    atoms_list = []

    while res/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < phi.shape[1]:
        vect_c = phi.T.dot(vect_y - phi.dot(vect_sparse))
        i_0 = np.argmax(np.abs(vect_c))
        atoms_list.append(i_0)
        vect_sparse[i_0] += vect_c[i_0]

        # Orthogonal projection.
        index = np.where(vect_sparse)[0]
        vect_sparse[index] = np.linalg.pinv(phi[:, index]).dot(vect_y)
        res = np.linalg.norm(vect_y - phi.dot(vect_sparse))

    return vect_sparse, atoms_list

# %% K-SVD TOOLS

def ksvd(Data, num_atoms, sparsity, initial_D=None, maxiter=10, etol=1e-10, approx=False, debug=True):
    # **implemented using column major order**
    Data = Data.T

    assert Data.shape[1] > num_atoms # enforce this for now

    # intialization
    if initial_D is not None: 
        D = initial_D / np.linalg.norm(initial_D, axis=0)
        Y = Data
        X = np.zeros([num_atoms, Data.shape[1]])
    else:
        # randomly select initial dictionary from data
        idx_set = range(Data.shape[1])
        idxs = np.random.choice(idx_set, num_atoms, replace=False)    
        Y = Data[:,np.delete(idx_set, idxs)]
        X = np.zeros([num_atoms, Data.shape[1] - num_atoms])
        D = Data[:,idxs] / np.linalg.norm(Data[:,idxs], axis=0)

    # repeat until convergence or stopping criteria
    error_norms = []
    
    # iterator = tqdm(range(1,maxiter+1)) if debug else range(1,maxiter+1)
    # for iteration in iterator:
    for iteration in range(maxiter):
        # sparse coding stage: estimate columns of X
        gram = (D.T).dot(D)
        Dy = (D.T).dot(Y)
        X = orthogonal_mp_gram(gram, Dy, n_nonzero_coefs=sparsity)
        # X = omp(gram, Dy)
        # codebook update stage
        for j in range(D.shape[1]):
            # index set of nonzero components
            index_set = np.nonzero(X[j,:])[0]
            if len(index_set) == 0:
                # for now, replace with some white noise
                if not approx:
                    D[:,j] = np.random.randn(*D[:,j].shape)
                    D[:,j] = D[:,j] / np.linalg.norm(D[:,j])
                continue
            # approximate K-SVD update
            if approx:
                E = Y[:,index_set] - D.dot(X[:,index_set])
                D[:,j] = E.dot(X[j,index_set])     # update D
                D[:,j] /= np.linalg.norm(D[:,j])
                X[j,index_set] = (E.T).dot(D[:,j]) # update X
            else:
                # error matrix E
                E_idx = np.delete(range(D.shape[1]), j, 0)
                E = Y - np.dot(D[:,E_idx], X[E_idx,:])
                U,S,VT = np.linalg.svd(E[:,index_set])
                # update jth column of D
                D[:,j] = U[:,0]
                # update sparse elements in jth row of X    
                X[j,:] = np.array([
                    S[0]*VT[ 0,np.argwhere(index_set==n)[0][0] ]
                    if n in index_set else 0
                    for n in range(X.shape[1])])
        # stopping condition: check error        
        err = np.linalg.norm(Y-D.dot(X),'fro')
        error_norms.append(err)
        if err < etol:
            break
    return D,X, np.array(error_norms)


# %% K-SVD APPLICATIONS

def denoise_ksvd(noisy_image, learning_image, window_shape, window_step, learning_ratio, tol=1e-6, max_iterations=1):
    # Prepare dataset from image patches
    padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')
    noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, window_shape, window_step)
    padded_lea_image = pad(learning_image, pad_width=window_shape, mode='symmetric')
    lea_patches, lea_patches_shape = patch_matrix_windows(padded_lea_image, window_shape, window_step)

    # Initialise dictionary
    k = int(learning_ratio*lea_patches.shape[1])
    indexes = np.random.random_integers(0, lea_patches.shape[1]-1, k)

    basis = lea_patches[:, indexes]
    basis /= np.sum(basis.T.dot(basis), axis=-1)

    # Run K-SVD
    basis_final, sparse_final, _ = ksvd(noisy_patches, 100, 60)

    # Reconstruct image
    patches_approx = basis_final.dot(sparse_final)
    padded_denoised_image = image_reconstruction_windows(padded_noisy_image.shape, patches_approx, noisy_patches_shape, window_step)

    shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
    denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]
    return denoised_image

def denoise(noisy_image, num_atoms, sparsity, patch_size):
    # patch_size = (8,8)
    patches = extract_patches_2d(noisy_image, patch_size)
    data = np.resize(patches, (patches.shape[0], patch_size[0]*patch_size[1]))

    dictionary, sparse_vecs = ksvd(data, num_atoms, sparsity)

    denoised_image = dictionary.dot(sparse_vecs)
    denoised_image = np.resize(denoised_image, (patches.shape[0], patch_size[0], patch_size[1]))

    return reconstruct_from_patches_2d(denoised_image, noisy_image.shape)
# %%
import sys
import numpy as np

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from functions_ksvd import *

# %% INITIALISE SETTINGS
window_shape = (8,8)
step = 4
ratio = 1
tol = 1e-6
noise_std = 10
max_iterations = 5

image = io.imread('Images/house.tif', 0)
learning_image = image
m,n = image.shape

noisy_image = image + noise_std*np.random.randn(m,n)

# %% DENOISE USING K-SVD
denoised_image = denoise(noisy_image, 1000, 16, window_shape)

# %% PLOTS
style.use('classic')
rcParams['text.usetex'] = True
rcParams.update({'font.size': 10})
rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}'] 

fig, plts = plt.subplots(1,3,figsize=(10,6))
plts[0].imshow(image, vmin=0, vmax=255, cmap='gray')
plts[0].set_title(r"Original Image")

plts[1].imshow(noisy_image, vmin=0, vmax=255, cmap='gray')
plts[1].set_title(r"Noisy Image, $\sigma=%.2f$"%(noise_std))

plts[2].imshow(denoised_image, vmin=0, vmax=255, cmap='gray')
plts[2].set_title(r"Denoised Image using K-SVD")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_4/Answers/figures/disimages.eps', format='eps')
plt.show()

# %%

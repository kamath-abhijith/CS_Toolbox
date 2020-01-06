import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib import style

## Load signal and dictionary
load_dict_f = loadmat('hw1problem3.mat')

Psi = load_dict_f['Psi']
f = load_dict_f['f']

## Plots
style.use('ggplot')
style.use('dark_background')

plt.figure()
plt.plot(f,color='red')
plt.show()
import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib import style

from matchingPursuit import *

## Load signal and dictionary
load_dict_f = loadmat('hw1problem3.mat')

Psi = load_dict_f['Psi']
f = load_dict_f['f']

sparse_code = mat_pursuit(Psi,f,32)
f_rec = Psi.dot(sparse_code)

## Plots
style.use('ggplot')
style.use('dark_background')

fig, plts = plt.subplots(2)
plts[0].plot(f,color='green')
plts[0].plot(f_rec,'--',color='red')
plts[1].stem(sparse_code,'-b')
plt.show()
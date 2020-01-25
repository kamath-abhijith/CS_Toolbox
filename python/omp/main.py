import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from matchingPursuit import *

## Load signal and dictionary
load_dict_f = loadmat('hw1problem3.mat')

A = load_dict_f['Psi']
b = load_dict_f['f']

x_mp = greedy_mp(A,b,32)
b_rec_mp = A.dot(x_mp)

x_omp = greedy_omp(A,b,32).real
b_rec_omp = A.dot(x_omp)

## Error metrics
eb_mp = b-b_rec_mp
eb_mp_norm = np.linalg.norm(eb_mp)
eb_mp_norm_db = 20*np.log10(eb_mp_norm)
print("l2 norm of the error using MP: ",eb_mp_norm_db,"dB")

eb_omp = b-b_rec_omp
eb_omp_norm = np.linalg.norm(eb_omp)
eb_omp_norm_db = 20*np.log10(eb_omp_norm)
print("l2 norm of the error using OMP: ",eb_omp_norm_db,"dB")

## Plots
style.use('ggplot')
# style.use('dark_background')

rcParams['text.usetex'] = True

fig, plts = plt.subplots(2,2)
plts[0][0].plot(b,color='green',label='True')
plts[0][0].plot(b_rec_mp,'--',color='red',label='Estimate')
plts[0][0].set_xlabel(r"m")
plts[0][0].set_ylabel(r"b")
plts[0][0].set_title(r"Matching Pursuit")
plts[0][0].legend()
plts[1][0].stem(x_mp.real,'-b')
plts[1][0].set_xlabel(r"N")
plts[1][0].set_ylabel(r"x")

plts[0][1].plot(b,color='green',label='True')
plts[0][1].plot(b_rec_omp,'--',color='red',label='Estimate')
plts[0][1].set_xlabel(r"m")
plts[0][1].set_ylabel(r"b")
plts[0][1].set_title(r"Orthogonal Matching Pursuit")
plts[0][1].legend()
plts[1][1].stem(x_omp.real,'-b')
plts[1][1].set_xlabel(r"N")
plts[1][1].set_ylabel(r"x")
plt.show()
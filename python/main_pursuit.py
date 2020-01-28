import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from cs_pkg import *

## Load signal and dictionary
load_dict_f = loadmat('hw1problem3.mat')

A = load_dict_f['Psi']
b = load_dict_f['f']

x_omp = greedy_omp(A,b,32).real
b_rec_omp = A.dot(x_omp)

x_ista = pursuit_ista(A,b,0.01,1000)
b_rec_ista = A.dot(x_ista)

## Error metrics
eb_ista = b-b_rec_ista
eb_ista_norm = np.linalg.norm(eb_ista)
eb_ista_norm_db = 20*np.log10(eb_ista_norm)
print("l2 norm of the error using ISTA: ",eb_ista_norm_db,"dB")

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
plts[0][0].plot(b_rec_ista,'--',color='red',label='Estimate')
plts[0][0].set_xlabel(r"m")
plts[0][0].set_ylabel(r"b")
plts[0][0].set_title(r"ISTA")
plts[0][0].legend()
plts[1][0].stem(x_ista.real,'-b')
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
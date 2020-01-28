import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from cs_pkg import *

## Time support
n = np.arange(512)
N = len(n)
t = np.arange(0,1,1.0/N)

k = 3
f = np.array([2,4,10])
a = np.random.rand(k)
theta = 2*np.pi*np.random.rand(k)

y = np.zeros(N)
for i in range(k):
	y += a[i]*np.sin(2*np.pi*f[i]*n/N+theta[i])

# y *= np.hanning(N)

## Sensing matrix
m = 30
phi = np.random.randn(m,N)

# Measurements
b = phi.dot(y)
b = b[:,np.newaxis]

# Measurement matrix
F = np.fft.fft(np.identity(N,dtype=float))
A = phi.dot(np.conj(F.transpose()))

x = (F.dot(y))

## Sparse recovery
x_rec = (greedy_omp(A,b,m,1e-12))

# Reconstruction
y_rec = np.conj(F.transpose()).dot(x_rec)

# Error metrics
ey = y-y_rec[:,0]
ey_norm = np.linalg.norm(ey)
ey_norm_db = 20*np.log10(ey_norm)
print("l2 norm of the error: ",ey_norm_db)

## Plots
style.use('ggplot')
style.use('dark_background')

rcParams['text.usetex'] = True

plt.figure()
plt.plot(t,y,'-',color='green',label='True')
plt.plot(t,y_rec,'--',color='red',label='Recovered')
plt.ylabel(r'$y(t)$')
plt.xlabel('time')
plt.title('Time Domain Signal')
plt.legend()

fig, plts = plt.subplots(2,sharex=True)
plts[0].stem(x.real,'g',markerfmt='gx',label='True')
plts[0].set_ylabel(r'$\mathbf{x}$')
plts[0].set_xlabel(r'$N$')
plts[1].stem(x_rec.real,'r',markerfmt='rx',label='Estimate')
plts[1].set_ylabel(r'$\mathbf{\bar{x}}$')
plts[1].set_xlabel(r'$N$')
plt.suptitle('Sparsity in Fourier Domain')
plt.show()
# Compressed Sensing Toolbox
This repository contains MATLAB/Python routines used in Compressed Sensing.

## Useful Resources
- [Justin Romberg's short course](http://jrom.ece.gatech.edu/tsinghua-oct13/)
- Foucart's book: A Mathematical Introduction to Compressive Sensing

## Contents
**Greedy Algorithms**
- OMP
	- Matching Pursuit
	- Orthogonal Matching Pursuit

**Basis Pursuit**
- LASSO using ADMM
	- ADMM Solver for LASSO

- LASSO using ISTA
	- ISTA Solver for LASSO
	- Fast ISTA Solver for LASSO, FISTA

- LASSO using IRLS
	- IRLS solver for LASSO

**L2-norm Constraint**
- Tikhonov Regularization
	- Closed form Tikhonov solver using SVD
	- Tikhonov solver using Majorizer-Minimization

_**Applications**_
- dct2
	- Sparse recovery algorithms on images built on a complete 2D-DCT basis
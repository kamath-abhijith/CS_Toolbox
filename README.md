# Compressed Sensing Toolbox
This repository contains MATLAB routines used in Compressed Sensing that could be used for other projects. This repository is most likely not the best as it is created whilst learning those algorithms. The programmes are not the most optimal and the functions might not be reusable. However, the programmes work as expected and could be used for proof of concept.

This repository also includes some MATLAB programmes implementing optimization algorithms some of which are used in Compressed Sensing and the rest elsewhere. This repository also contains the assignments in [Justin Romberg's short course](http://jrom.ece.gatech.edu/tsinghua-oct13/) on Compressed Sensing at Tsinghua.


## Contents
**Greedy algorithms**
- omp
	- Matching Pursuit
	- Orthogonal Matching Pursuit

**Convex-relaxed sparsity constraint**
- lasso_admm
	- ADMM Solver for LASSO

- lasso_ista
	- ISTA Solver for LASSO
	- Fast ISTA Solver for LASSO, FISTA

- irls
	- IRLS solver for LASSO

**Low energy constraint**
- tikhonov
	- Closed form Tikhonov solver using SVD
	- Tikhonov solver using Majorizer-Minimization
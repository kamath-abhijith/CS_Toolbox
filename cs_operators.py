'''

OPERATORS FOR COMPRESSIVE SENSING AND SPARSE SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% IMPORT LIBRARIES

import numpy as np

# %% SAMPLING OPERATORS

def random_sampling(m, N):
    ''' m x N random sampling matrix '''
    return np.sqrt(1/m)*np.random.randn(m,N)
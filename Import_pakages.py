import numpy as np
import matplotlib.pyplot as plt

from cmcrameri import cm
import matplotlib.colors as colors
cmp = cm.devon
cmp = cmp.reversed()

import lmfit as lmfit

import scipy as scipy
from scipy.io import savemat
from scipy.io import loadmat

me = (0.511*1e6)/(299792458 *1e10)**2  # eV s^2 Angstrom^-2
hbar = 6.582*1e-16   # eV s

#######################################################################################################################

def find_value_index(matrix, y):
    # Ensure y is an array and reshaped for broadcasting
    # matrix shape: (M, N), y shape: (K, 1, 1)
    y = np.atleast_1d(y)[:, np.newaxis, np.newaxis]
    
    # Calculate absolute difference across all values of y
    # Then find the argmin across the matrix dimensions (axes 1 and 2)
    diff = np.abs(matrix - y)
    
    # Flatten the matrix dimensions to find the global minimum index per 'y'
    indices = diff.reshape(len(y), -1).argmin(axis=1)
    
    return indices
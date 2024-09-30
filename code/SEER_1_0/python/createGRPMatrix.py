import numpy as np
from scipy.linalg import orth

# Create a Gaussian Random Projection matrix PP of size(nInDIms, nOutDims).
#
# nInDims ... number of input dimensions (rows in PP)
# nOutDims ... number of output dimensions (columns in PP)
# seed ... is the seed for the random number generator
#
# If nInDIms==nOutDims, the output is orthogonalized (this takes some time
# for large dimensions).
#
# peer.neubert@etit.tu-chemnitz.de, 2022

def create_grpmatrix(n_in_dims, n_out_dims, seed):
    np.random.seed(seed)
    
    if n_in_dims == n_out_dims:
        PP = np.random.randn(int(n_in_dims * 1.2), n_out_dims).astype(np.float32)
        PP = orth(PP.T).T
        assert PP.shape[0] == n_in_dims
    else:
        PP = np.random.randn(n_in_dims, n_out_dims).astype(np.float32)
    
    PP = normalize_columns(PP)
    
    return PP

def normalize_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms

# Beispielnutzung:
n_in_dims = 100
n_out_dims = 50
seed = 42

PP = create_grpmatrix(n_in_dims, n_out_dims, seed)
print(PP)
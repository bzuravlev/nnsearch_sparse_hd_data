import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth
import matplotlib.pyplot as plt

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

def run_seer(M, data, add_new_exemplars, d_M, k, lambda_val):
    # Placeholder implementation of runSEER function
    # This should be replaced with the actual SEER algorithm
    return M, data

def create_pr(S, GT_hard, GT_soft):
    if GT_soft is not None and np.any(GT_soft):
        S[np.logical_and(GT_soft, ~GT_hard)] = np.min(S)

    GT = GT_hard.astype(bool)
    
    R = [0]
    P = [1]
    
    start_v = np.max(S)
    end_v = np.min(S)
    
    for i in np.linspace(start_v, end_v, 100):
        B = S >= i
        
        TP = np.sum(np.logical_and(GT, B))
        FN = np.sum(np.logical_and(GT, ~B))
        FP = np.sum(np.logical_and(~GT, B))
        
        P.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        R.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    
    return P, R

def normr(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

def demo_seer():
    """
    Demonstration of SEER on an example dataset. The actual SEER
    implementation is in runSEER.m

    peer.neubert@etit.tu-chemnitz.de, 2022
    """
    
    print('Running on Gardens Point Walking day_left vs. night_right')
    database_path = 'data/HDC_DELF_GardensPointWalking_day_left.mat'
    query_path = 'data/HDC_DELF_GardensPointWalking_night_right.mat'
    gt_path = 'data/groundTruth_GPW_DL_NR.mat'
    n_in_dims = 4096  # dimensionality of the input descriptors
    
    # prepare projection
    print('Prepare projection matrix (needs only be computed once)')
    PP = create_grpmatrix(n_in_dims, 4096, 0)
    
    # parameters
    k = 50
    lambda_val = 2
    d_M = 200
    
    # process database
    print('Load database')
    DB = loadmat(database_path)['Y']
    DB = DB @ PP  # project
    DB_mean = np.mean(DB, axis=0)
    DB_Y = DB - DB_mean  # standardize
    
    # run SEER
    M = []
    np.random.seed(873734)
    print('SEER: first run on database')
    M, DB_SEER_run1 = run_seer(M, DB_Y, 1, d_M, k, lambda_val)
    
    # second run to generate output descriptor
    print('SEER: second run on database')
    _, DB_SEER_run2 = run_seer(M, DB_Y, 0, d_M, k, lambda_val)
    
    # process query
    print('Load query')
    Q = loadmat(query_path)['Y']
    Q = Q @ PP  # project
    Q_Y = Q - DB_mean  # standardize with DB mean
    
    # run SEER, generates output descriptor in a single run
    print('SEER: run on query')
    _, Q_SEER = run_seer(M, Q_Y, 0, d_M, k, lambda_val)
    
    # evaluate
    S = normr(DB_SEER_run2) @ normr(Q_SEER).T  # this is a sparse matrix
    
    GT = loadmat(gt_path)['GT']
    P, R = create_pr(S, GT['GThard'], GT['GTsoft'])
    AUC = np.trapz(R, P)
    
    print(f'The area under the precision-recall curve is: {AUC:.4f}')

# Beispielnutzung:
demo_seer()
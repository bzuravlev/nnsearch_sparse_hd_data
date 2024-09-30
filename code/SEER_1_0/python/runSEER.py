import numpy as np
from scipy.sparse import lil_matrix
import scipy.io as sio

#Implementation of Sparse Exemplar Ensemble Representations (SEER) 

#Peer Neubert and Stefan Schubert (2022), "SEER: Unsupervised and 
#sample-efficient environment specialization of image descriptors", 
#Proceeding of Robotics: Science And Systems, New York, USA

#This is a batch version that computes output descriptors for a batch of
#input descriptors Y.

#M ... sparse exemplar memory, each COLUMN is an exemplar, can be empty. It
#          is important to allocate the appropriate amount of memory. If an
#          empty matrix is provided, then the memory is allocated within this
#          function.
#Y ... input descriptor, each row is a descriptor. Should be a distributed 
#          representation.
#updatedM_flag ... toggles whether new rows/exemplars are added to M
#d_M ... number of non-zero elements in the exemplars in M, default: 200
#k ... minimum number of exemplars per database descriptor, default: 50
#lambda ... factor on k for non-zero elements in the output descriptor, default: 2

#M ... see input, will be updated if updatedM_flag is set
#DS ... sparse output descriptor

#peer.neubert@etit.tu-chemnitz.de, 2022

def run_seer(M, Y, update_M_flag, d_M=200, k=50, lambda_val=2):
    
    # Allocate memory for output descriptor
    DS = lil_matrix((Y.shape[0], M.shape[1] if M is not None else 0), dtype=np.float32)
    
    # Allocate memory for M if it is empty AND there has not already been allocated some memory
    if M is None or M.shape[1] == 0:
        M = lil_matrix((Y.shape[1], 1), dtype=np.float32)
    
    print(M.shape)
    # L2 normalize each input descriptor
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Process each descriptor
    for i in range(Y.shape[0]):
        if i % 100 == 0:
            print(f'  running image {i+1} of {Y.shape[0]} ({M.shape[1]} exemplars) ')
        
        # Compute similarity
        if M.shape[1] > 0:
            S = Y[i, :] @ M.toarray()
        else:
            S = np.array([])
        
        # Sparsify
        if S.size > 0:
            knn_sim = np.partition(S, -lambda_val*k)[-lambda_val*k:]
            knn_idx = np.argpartition(S, -lambda_val*k)[-lambda_val*k:]
        else:
            knn_sim = np.array([])
            knn_idx = np.array([])
        
        if update_M_flag:
            # Apply threshold
            expected_similarity = d_M / Y.shape[1]
            thresh_idx = knn_sim >= expected_similarity
            knn_idx_above_thresh = knn_idx[thresh_idx]
            print("reached line 67")
        
            # Potentially create new patterns in M        
            if knn_idx_above_thresh.size < k:
                n_new_patterns = k - knn_idx_above_thresh.size
                PI, PV = create_exemplars(Y[i, :], d_M, n_new_patterns)
                print("PI shape:")
                print(PI.shape)
                new_pattern_idx = np.arange(M.shape[1], M.shape[1] + PI.shape[0])
                new_pattern_activity = np.sum(PV * PV, axis=1)
                print("reached line 75")
                for p_idx in range(PI.shape[0]):
                    M = np.hstack((M, lil_matrix((Y.shape[1], 1), dtype=np.float32)))
                    M[PI[p_idx], -1] = PV[p_idx]
            else:
                new_pattern_idx = np.array([])
                new_pattern_activity = np.array([])
                n_new_patterns = 0
        else:
            new_pattern_idx = np.array([])
            new_pattern_activity = np.array([])
            n_new_patterns = 0
        
        # Output descriptor is kNN results above threshold and new patterns
        n_remaining_kNN = min(knn_idx.size, k * lambda_val - n_new_patterns)
        descr_idx = np.hstack((knn_idx[:n_remaining_kNN], new_pattern_idx))
        descr_activity = np.hstack((knn_sim[:n_remaining_kNN], new_pattern_activity))
                
        # Create output descriptor
        DS[i, descr_idx] = descr_activity
    
    return M, DS

def create_exemplars(input_Y, n_dim_samples, n_patterns):
    """
    Sample new exemplars

    input_Y ... continuous input descriptor
    PI ... index matrix, each row is vector of size nDimSamples
    PV ... the values from inputY that correspond to these indexes 
    """
    PI = np.zeros((n_patterns, n_dim_samples), dtype=int)
    for i in range(n_patterns):
        if n_dim_samples == input_Y.shape[0]:
            PI[i, :] = np.arange(input_Y.shape[0])
        else:
            PI[i, :] = np.random.choice(input_Y.shape[0], n_dim_samples, replace=False, p=np.abs(input_Y)/np.sum(np.abs(input_Y)))
    
    PV = input_Y[PI]
    return PI, PV

if __name__=="__main__":
    DB = sio.loadmat('../data/HDC_DELF_GardensPointWalking_day_left.mat')['Y']
    Q = sio.loadmat('../data/HDC_DELF_GardensPointWalking_night_right.mat')['Y']
    gt = sio.loadmat('../data/groundTruth_GPW_DL_NR.mat')['GT']
    M = None
    M, DS = run_seer(M, DB, True, 200, 50, 2)
    _, Q_SEER = run_seer(M, Q, False, 200, 50, 2)
    
# Beispielnutzung:
# import scipy.io as sio
# DB = sio.loadmat('data/HDC_DELF_GardensPointWalking_day_left.mat')['Y']
# Q = sio.loadmat('data/HDC_DELF_GardensPointWalking_night_right.mat')['Y']
# gt = sio.loadmat('data/groundTruth_GPW_DL_NR.mat')['GT']
# M = None
# M, DS = run_seer(M, DB, True, 200, 50, 2)
# _, Q_SEER = run_seer(M, Q, False, 200, 50, 2)
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize

class seer:

    def __init__(self):
        return
        
    def compute_M_and_Y(self, M, X, d_M=200, k=50, lambda_val=2):
        #Allocate memory for output descriptor
        DS = lil_matrix((0, 0), dtype=np.float32)
        #L2 normalize each input descriptor
        X = normalize(X, axis=1, norm='l2')
        mat_empty = True
        expectedSim = d_M / X.shape[1]
        print("expectedSim: ", expectedSim)
        for i in range(X.shape[0]):
            if i % 10 == 0:
                n_ex = 0
                if M is not None:
                    n_ex = M.shape[1]
                print(f'  running image {i+1} of {X.shape[0]} ({n_ex} exemplars) ')
            c = 0
            if M is not None:
                S = X[i] @ M
                mat_empty = False
                c = np.count_nonzero(S > expectedSim)
            else:
                S = np.array([])
                mat_empty = True
            #sample k-c new exemplars
            if k > c:
                PI,PV = self.create_exemplars(X[i, :], d_M, k-c)
                if mat_empty:
                    M = np.empty((X.shape[1],0))
                    
                #append empty columns to M
                new_cols = np.zeros((X.shape[1],k-c), dtype=np.float32)
                M = np.hstack([M, new_cols])
                for p_idx in range(k-c, 0, -1):
                    for p_idy in range(PI.shape[0]):
                        #fill new columns with values
                        #use deepcopy
                        M[ PI[p_idy, -p_idx], -p_idx] = np.copy(PV[p_idy, -p_idx])
                
                mat_empty = False
                #append to S
                for index in range(k-c, 0, -1):
                    #compute and append missing values
                    new_val = X[i] @ M[:,-index]
                    S = np.append(S, new_val)
            #sparsify
            S= self.keep_n_largest(S, lambda_val*k)
            DS= self.pad_and_append_SEER(DS,S)
        return M, DS

    
    def keep_n_largest(self, arr, n):
        #flat array for simple manipulation
        flat_arr = arr.flatten()
        #find threshold for n greatest values
        if n >= len(flat_arr):
            threshold = np.min(flat_arr)
        else:
            threshold = np.partition(flat_arr, -n)[-n]
        
        #create mask which is True for n greatest values
        mask = arr >= threshold
        #create new array using this mask to set lower values to zero
        result = np.where(mask, arr, 0)
        
        return result


    def pad_and_append_SEER(self, D, new_row):
        if D.shape[1] > 0:
            #Get the number of columns in the existing matrix and the new row
            num_cols_M = D.shape[1]
            num_cols_new = new_row.size
            
            #Pad the existing matrix with zeros to match the number of columns in the new row
            if num_cols_M < num_cols_new:
                padding_M = sp.lil_matrix((D.shape[0], num_cols_new - num_cols_M), dtype=D.dtype)
                D = sp.hstack([D, padding_M])
            
            #Convert the new row to a sparse matrix
            new_row_sparse = sp.lil_matrix(new_row)
            
            #Append the new row to the matrix D in place
            D = sp.vstack([D, new_row_sparse])
        else:
            #Convert the new row to a sparse matrix and assign it to D
            D = sp.lil_matrix(new_row)
        
        return D


    def create_exemplars(self, input_Y, n_dim_samples, n_patterns):
        PI = np.zeros((n_dim_samples, n_patterns), dtype=int)
        PV = np.zeros((n_dim_samples, n_patterns), dtype=int)
        for i in range(n_patterns):
            if n_dim_samples == input_Y.shape[0]: # prüfen!
                PI[:, i] = np.arange(input_Y.shape[0]) #prüfen!
            else:
                PI[:, i] = np.random.choice(input_Y.shape[0], n_dim_samples, replace=False, p=np.abs(input_Y)/np.sum(np.abs(input_Y)))
        
        PV = input_Y[PI]
        
        #sparsify
        for col in range(PV.shape[1]):
            PV[:,col] = self.keep_n_largest(PV[:,col], 200)
        return PI, PV


    def compute_seer_descriptors(self, input_Y, M, lambda_val, k):
        # Allocate memory for output descriptor
        DS = lil_matrix((input_Y.shape[0], M.shape[1]), dtype=np.float32)
        # L2 normalize each input descriptor
        input_Y = input_Y / np.linalg.norm(input_Y, axis=1, keepdims=True)
        for i in range(input_Y.shape[0]):
            S = input_Y[i] @ M
            S = self.keep_n_largest(S, lambda_val*k)
            DS[i] = S
        return DS
    
    def save_matrix_to_file(self, data, file):
        conv_data = sp.csr_matrix(data)
        sp.save_npz(file, conv_data)

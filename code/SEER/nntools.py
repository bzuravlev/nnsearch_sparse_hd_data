import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from collections import Counter
from scipy.spatial import distance
from time import perf_counter

def print_gs_matrix(matrix):
    plt.imshow(matrix, cmap='gray')
    plt.colorbar()
    plt.show()
    
def print_array_scatter(array, title, ylabel):
    X = np.arange(0, array.shape[0])

    plt.scatter(X, array, color='r', s=3, label=title)

    plt.xlabel("Image")
    plt.ylabel(ylabel)
    plt.title("Results for NN Search")

    plt.legend()
    plt.show()
    
def get_position_of_correct_frame(matrix, similarity_mode):
    length = matrix.shape[0]
    positions = np.zeros(length)
    #input: similarity matrix
    for i in range(length):
        sorted_indices = np.argsort(matrix[i])
        #similarity or distance mode
        if similarity_mode:
            pos = length - np.where(sorted_indices==i)[0][0] - 1
        else:
            pos = np.where(sorted_indices==i)[0][0]
        positions[i] = pos
    return positions

def calc_hard_matrix(matrix, similarity_mode):
    #Create a new matrix with the same shape, filled with zeros
    result = np.zeros_like(matrix)
    if similarity_mode:
        indices = np.argmax(matrix, axis=1)
    else:
        indices = np.argmin(matrix, axis=1)
    result[np.arange(matrix.shape[0]), indices] = 1
    #return a matrix where the best candidate per row is set to one
    return result

def dist_principal_diagonal(matrix):
    length = matrix.shape[0]
    differences = np.zeros(length)
    #use the fact, that we need a principal diagonal here
    for i in range(length):
        difference = np.abs(np.argmax(matrix[i]) - i)
        differences[i] = difference
    return differences

def get_entries_per_dimension(sparse_mat):
    num = sparse_mat.shape[0]
    res = np.zeros(num)
    for i in range(num):
        res[i] = sp.csr_matrix.count_nonzero(sparse_mat[:,i])
    return res

def lk_distance(k, x1, x2):
    #test if x1 and x2 have equal length
    if (len(x1) != len(x2)):
        print('Input vectors have different lengths!')
        return
    else:
        #compute and return lk length
        result = np.sum(np.abs(x1 - x2) ** k) ** (1/k)
        return result

def sequential_lk_dist(Xin, Yin, k):
    result_Matrix = np.empty((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
    for i in range(Xin.shape[0]):
        for j in range(Yin.shape[0]):
            dist = lk_distance(k, Xin[i].toarray()[0], Yin[j].toarray()[0])
            result_Matrix[j][i] = dist
    return result_Matrix

def compute_matching_dimensions_sequential(Xin, Yin):
        result_Matrix = np.zeros((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
        for i in range(Xin.shape[0]):
            for j in range(Yin.shape[0]):
                #get number of matching dimensions
                matches = np.size(get_common_indices(Xin[i], Yin[j]))
                result_Matrix[j][i] = matches
        return result_Matrix
    
def build_index_structure_matching_dims(Q_Data):
    indices_lists = [[] for _ in range(Q_Data.shape[1])]
    for i in range(Q_Data.shape[0]):
        vec = Q_Data[i]
        #get populated dimensions
        used_dims = vec.nonzero()[1]
        for dim in used_dims:
            indices_lists[dim].append(i)
    return indices_lists

def compute_matching_dimensions_w_index(Xin, Yin):
        result_Matrix = np.zeros((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
        indices_lists = build_index_structure_matching_dims(Yin)
        #dtype=object for lists
        indices_array = np.array(indices_lists, dtype=object)  
        for i in range(Xin.shape[0]):
            #get nonzero Elements of descriptor
            nonzero_dims = Xin[i].nonzero()[1]
            #get all candidate indices
            candidates_ind = indices_array[nonzero_dims]
            #combine lists and count occurences of every index
            combined_list = [item for sublist in candidates_ind for item in sublist]
            counter = Counter(combined_list)
            #write occurences to correct place
            for index, count in counter.items():   
                result_Matrix[index][i] = count
        return result_Matrix
       
def get_common_indices(sparsem1, sparsem2):
    #Get and return the indices of the dimensions which are present in both sparse vectors
    return np.intersect1d(sparsem1.nonzero()[1],  sparsem2.nonzero()[1])

def get_values_on_common_dimension(sparsem1, sparsem2):
    #Get common indices
    common_indices = get_common_indices(sparsem1, sparsem2)
    
    #Get and return corresponding values
    common_values1 = sparsem1[0, common_indices].toarray()[0]
    common_values2 = sparsem2[0, common_indices].toarray()[0]
    return common_values1, common_values2

def distance_on_matching_dims(sparsem1, sparsem2, k):
    common_values1, common_values2 = get_values_on_common_dimension(sparsem1, sparsem2)
    if common_values1.size == 0:
        return np.inf
    else:
        #compute and return distance on common distances
        result = lk_distance(k, common_values1, common_values2)
        return result

def distance_on_matching_dims_normalized(sparsem1, sparsem2, k):
    common_values1, common_values2 = get_values_on_common_dimension(sparsem1, sparsem2)
    if common_values1.size == 0:
        return np.inf
    else:
        result = lk_distance(k, common_values1, common_values2) / common_values1.size
        return result

def compute_dist_on_matching_dimensions_sequential(Xin, Yin, k):
    result_Matrix = np.full((Xin.shape[0], Yin.shape[0]), np.inf, dtype=np.float32)
    for i in range (Xin.shape[0]):
        matrix1 = Xin[i]
        for j in range (Yin.shape[0]):
            matrix2 = Yin[j]
            dist = distance_on_matching_dims(matrix1, matrix2, k)
            result_Matrix[j][i] = dist
    return result_Matrix

def compute_dist_on_matching_dimensions_normalized_sequential(Xin, Yin, k):
    result_Matrix = np.full((Xin.shape[0], Yin.shape[0]), np.inf, dtype=np.float32)
    for i in range (Xin.shape[0]):
        matrix1 = Xin[i]
        for j in range (Yin.shape[0]):
            matrix2 = Yin[j]
            dist = distance_on_matching_dims_normalized(matrix1, matrix2, k)
            result_Matrix[j][i] = dist
    return result_Matrix

def get_lowest_per_dimension(matrix):
    #convert all zeros to infinity
    arr_with_inf = np.where(matrix.toarray() == 0, np.inf, matrix.toarray())
    #get lowest per dimension while ignoring zeros
    lowest_per_dim = np.amin(arr_with_inf, axis=0)
    return lowest_per_dim

def piDist_simple_sequential(Xin, Yin, p):
    # iterate over all points
    result_Matrix = np.zeros((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
    highest_per_dim = np.amax(Xin, axis=0).toarray()[0]
    lowest_per_dim = get_lowest_per_dimension(Xin)
    range_per_dim = highest_per_dim - lowest_per_dim
    for i in range(Xin.shape[0]):
        matrix1 = Xin[i]
        for j in range(Yin.shape[0]):
            matrix2 = Yin[j]
            common_indices = get_common_indices(matrix1, matrix2)
            valuesX, valuesY = get_values_on_common_dimension(matrix1, matrix2)
            differences = np.abs(valuesX - valuesY)
            similarity = (np.sum( (1- differences / range_per_dim[common_indices])**p ))** (1/p)
            result_Matrix[j][i] = similarity
    return result_Matrix

def piDist_threshold_sequential(Xin, Yin, p, t):
    #initial similarity value is 0
    result_Matrix = np.zeros((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
    #get highest value in each dimension
    highest_per_dim = np.amax(Xin, axis=0).toarray()[0]
    #get lowest value in each dimension except for 0
    lowest_per_dim = get_lowest_per_dimension(Xin)
    #compute ranges in all dimensions
    range_per_dim = highest_per_dim - lowest_per_dim
    #iterate over all points
    for i in range(Xin.shape[0]):
        matrix1 = Xin[i]
        for j in range(Yin.shape[0]):
            matrix2 = Yin[j]
            common_indices = get_common_indices(matrix1, matrix2)
            valuesX, valuesY = get_values_on_common_dimension(matrix1, matrix2)
            #compute all differences on common dimensions
            differences = np.abs(valuesX - valuesY)
            #filter the values so that only close pairings are considered
            close_indices = np.where(differences < (range_per_dim[common_indices]*t))
            #compute the similarity value
            c_i_close = common_indices[close_indices]
            d_close = differences[close_indices]
            
            similarity = (np.sum( (1- d_close / range_per_dim[c_i_close])**p ))** (1/p)
            result_Matrix[j][i] = similarity
    return result_Matrix

def srp_lsh_sequential(Xin, Yin, num_rand_vectors):
    #initialize matrix with zeros
    result_Matrix = np.zeros((Xin.shape[0], Yin.shape[0]), dtype=np.float32)
    d = Xin.shape[1]
    mu = np.zeros(d)
    cov_matrix = np.eye(d)
    #create num_rand_vectors with mean 0 and an identity matrix as covariance matrix
    rand_vectors = np.random.multivariate_normal(mu, cov_matrix, num_rand_vectors)
    for i in range(Xin.shape[0]):
        vec1 = Xin[i].toarray()
        for j in range(Yin.shape[0]):
            #compute similarity
            vec2 = Yin[j].toarray()
            code1 = (np.dot(vec1, np.transpose(rand_vectors)) >= 0)[0]
            code2 = (np.dot(vec2, np.transpose(rand_vectors)) >= 0)[0]
            similarity = distance.hamming(code1, code2)
            result_Matrix[j][i] = similarity
    return result_Matrix


def print_matrices(gs_mat, dist_hard, rank, datastring, functionstring, *args):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5))
    s = args
    fig.suptitle(f"Dataset:  {datastring},  Function:  {functionstring}, Parameters: {s}")
    x = np.arange(0, gs_mat.shape[0])
    ax1.imshow(gs_mat, cmap='gray')
    ax1.set_title("Grayscale Results")
    
    values1, bins1, bars1 = ax2.hist(dist_hard, bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 200], edgecolor='white')
    ax2.bar_label(bars1, fontsize=10, color='navy')
    ax2.set_xscale('log')
    ax2.set_title("Distance to correct NN (frames)")
    
    values2, bins2, bars2 = ax3.hist(rank, bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 200], edgecolor='white')
    ax3.bar_label(bars2, fontsize=10, color='navy')
    ax3.set_xscale('log')
    ax3.set_title("Index of correct frame in sorted NNs")
    
    s1 = (f"_{s}_").replace('(','').replace(')','').replace(", ",'_').replace('.','-')
    dir = "./exports/"
    
    # Save the full figure
    file_full = (f"{dir}{functionstring}_{s1}_{datastring}_full").replace(' ', '')
    fig.savefig(file_full)

def run_and_evaluate_func_on_data(function, dataX, dataY, datastring, functionstring, similarity_mode, *args):
    start = perf_counter()
    #call function with given data and optional arguments
    res_matrix = function(dataX, dataY, *args)
    end = perf_counter()
    duration = end - start
    #create a matrix where only the best match is set to 1 and all others to 0
    res_matrix_dist_hard = dist_principal_diagonal(calc_hard_matrix(res_matrix, similarity_mode))
    #create a matrix where the entries are the indices of the sorted NNs
    res_rank = get_position_of_correct_frame(res_matrix, similarity_mode)
    print_matrices(res_matrix, res_matrix_dist_hard, res_rank, datastring, functionstring, *args)
    
    #create a mask for the top 5 findings for each descriptor 
    mask = res_rank < 5
    #Sum the boolean mask to get the count of elements less than 5
    res_matrix_count_tolerated = np.sum(mask)
    print(f"Duration {functionstring}: {duration} seconds")
    print(f"Median Distance to GT-NN: {np.median(res_matrix_dist_hard)}")
    print(f"Average Distance to GT-NN: {np.mean(res_matrix_dist_hard)}")
    print(f"Median Rank of GT-NN: {np.median(res_rank)}")
    print(f"Average Rank of GT-NN: {np.mean(res_rank)}")
    print(f"{res_matrix_count_tolerated} out of {res_matrix.shape[0]} within top 5 NNs")
    return res_matrix, res_matrix_dist_hard, res_rank
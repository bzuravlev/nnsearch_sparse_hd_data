import numpy as np


#Compute precision and recall vectors for a given similarity matrix
#and binary ground truth matrices.
    
#S ... similarity matrix
#GT_hard ... ground truth matching matrix: 1 at places that must be matched,
#            else 0; must have the same shape as S
#GT_soft ... ground truth places that CAN be matched without penalty; must
#            have the same shape as S
    
#P ... precision vector
#R ... recall vector
    
#peer.neubert@etit.tu-chemnitz.de, 2022
   

def create_pr(S, GT_hard, GT_soft):
    
    # Remove soft-but-not-hard entries
    if GT_soft is not None and np.any(GT_soft):
        S[np.logical_and(GT_soft, ~GT_hard)] = np.min(S)

    GT = GT_hard.astype(bool)  # Ensure logical datatype
    
    # Initialize precision and recall vectors
    R = [0]
    P = [1]
    
    # Select start and end threshold
    start_v = np.max(S)  # Start value for threshold
    end_v = np.min(S)  # End value for threshold
    
    # Iterate over different thresholds
    for i in np.linspace(start_v, end_v, 100):
        B = S >= i  # Apply threshold
        
        TP = np.sum(np.logical_and(GT, B))  # True positives
        FN = np.sum(np.logical_and(GT, ~B))  # False negatives
        FP = np.sum(np.logical_and(~GT, B))  # False positives
        
        P.append(TP / (TP + FP) if (TP + FP) > 0 else 0)  # Precision
        R.append(TP / (TP + FN) if (TP + FN) > 0 else 0)  # Recall
    
    return P, R

# Beispielnutzung:
S = np.array([[0.1, 0.4, 0.35], [0.8, 0.2, 0.7], [0.3, 0.9, 0.5]])
GThard = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
GTsoft = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])

P, R = create_pr(S, GThard, GTsoft)
print("Precision:", P)
print("Recall:", R)
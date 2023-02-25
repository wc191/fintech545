import numpy as np
from numpy import linalg

def simulate_pca(matrix, nsim, nval = None):
            
    # Eigenvalue decomposition
    vals, vecs = linalg.eigh(matrix)
    vals = vals.real
    vecs = vecs.real
    
    rank = vals.argsort()
    vals = vals[rank]
    vecs = vecs[:, rank]
    
    # sort from big to small
    flip = [i for i in range(len(vals)-1, -1, -1)]
    vals = vals[flip]
    vecs = vecs[:, flip]
    #print(vals)
    tv = sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if nval != None:
        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]

    print(f"Simulating with {len(posv)} PC Factors: {sum(vals)/tv*100:.2f}% total variance explained")
    B = np.dot(vecs, np.diag(np.sqrt(vals)))
    
    r = np.random.randn(len(vals), nsim)

    return np.dot(B, r)
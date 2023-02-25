import numpy as np
import scipy

def chol_psd(a):
    root = np.full(a.shape, 0.0)
    n = a.shape[1]
    # loop over columns
    for j in range(n):
        s = 0.0
        # if we are not on the first column, calculate the dot product of the preceeding row values.
        if j > 0:
            s =  root[j,:j] @ root[j,:j].T
        # Diagonal Element
        temp = a[j,j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j,j] = np.sqrt(temp)
        # Check for the 0 eigan value.  Just set the column to 0 if we have one
        if root[j,j] == 0.0:
            root[j,j:n-1] = 0.0
        else:
        # update off diagonal rows of the column
            for i in range(j+1, n):
                s = root[i,:j] @ root[j,:j].T
                root[i,j] = (a[i,j] - s) / root[j,j]
                
    return root

def direct_simulation(cov, mean = 0, explained_variance=1.0, samples_num=25000):
    L = chol_psd(cov)
    normal_samples = np.random.normal(size=(cov.shape[0], samples_num))
    samples = np.transpose(L @ normal_samples + mean)
    return samples

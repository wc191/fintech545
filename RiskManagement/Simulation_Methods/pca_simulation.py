import numpy as np
from numpy import linalg
from scipy.linalg import cholesky, eigvals
from ..Simulation_Methods.direct_simulation import chol_psd

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

def multivar_norm_simu(cov, method='direct', mean = 0, explained_variance=1.0, samples_num=25000):
    if method == 'direct':
        L = chol_psd(cov)
        normal_samples = np.random.normal(size=(cov.shape[0], samples_num))
        samples = np.transpose(L @ normal_samples + mean)
        return samples
    
    elif method == 'pca':
        vals, vecs = np.linalg.eigh(cov)
        idx = vals > 1e-8
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        if explained_variance == 1.0:
            explained_variance = (np.cumsum(vals)/np.sum(vals))[-1]
        
        n_components = np.where((np.cumsum(vals)/np.sum(vals))>= explained_variance)[0][0] + 1
        vecs = vecs[:,:n_components]
        vals = vals[:n_components]

        normal_samples = np.random.normal(size=(n_components, samples_num))
        
        B = vecs @ np.diag(np.sqrt(vals))
        samples = np.transpose(B @ normal_samples)
        
        return samples
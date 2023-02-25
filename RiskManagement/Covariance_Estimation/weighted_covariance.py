import numpy as np

def getweight(DF, L):
    
    nrows = DF.shape[0]
    weights = np.zeros(nrows)
    
    for i in range(nrows,0,-1):
        weights[i-1] = (1 - L) * (L**(i-1))
    weights = weights[::-1]
    return weights/weights.sum()

#weights = getweight(0.97,DR.shape[0])

def weightedcov(DF, L):
    
    matrix = np.matrix(DF.values)
    weights = getweight(DF, L)
    weightedcov = np.zeros([matrix.shape[1], matrix.shape[1]])
    
    for i in range(0,matrix.shape[1]):
        
        x = np.array(matrix[:,i]-matrix[:,i].mean())
        w_x = [weights[j]*x[j] for j in range(len(weights))]
        w_x = np.asmatrix(w_x)

        for k in range(0,matrix.shape[1]):
            
            y = np.asmatrix(matrix[:,k]-matrix[:,k].mean())
            weightedcov[i,k] += np.dot(w_x.T,y)
 
    return weightedcov

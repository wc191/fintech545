import pandas as pd
import numpy as np

def demean(DataFrame: pd.DataFrame, DateOrNot: str = 'Y'):
    
    nrows = DataFrame.shape[0]
    ncol = DataFrame.shape[1]
    Result = pd.DataFrame(np.zeros([nrows, ncol]))
    Result.index = DataFrame.index
    
    if DateOrNot == 'Y':
        # except the first one
        for i in range(1,ncol):
            Result.iloc[:,i] = DataFrame.iloc[:,i] - DataFrame.iloc[:,i].mean()
            
        Result.drop(Result.columns[0], axis=1, inplace=True)
        Result.insert(0, 'whatever', DataFrame.iloc[:,0]) 
        Result.columns = DataFrame.columns
        
    elif DateOrNot == "N":
        # except the first one
        for i in range(0,ncol):
            Result.iloc[:,i] = DataFrame.iloc[:,i] - DataFrame.iloc[:,i].mean()
            
        Result.columns = DataFrame.columns
    else:
        raise ValueError("you can only input Y or N")
    
    return Result
    
    
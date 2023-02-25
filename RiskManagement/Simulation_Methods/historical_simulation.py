import pandas as pd
import numpy as np

def get_exponential_weight(DataFrame :pd.DataFrame, L: float):
    
    nrows = DataFrame.shape[0]
    weights = np.zeros(nrows)
    
    for i in range(nrows,0,-1):
        weights[i-1] = (1 - L) * (L**(i-1))
        
    weights = weights[::-1]
    
    return weights/weights.sum()

def historical_simulation(DataFrame :pd.DataFrame, 
                          LastPrice: list = None, 
                          Holding: list = None,
                          DateOrNot: str = 'Y',
                          N: int = 10000, 
                          type: str = 'exponential', 
                          L: float = 0.97):
    
    # determine the weight
    if type == 'exponential':
        weight = get_exponential_weight(DataFrame, L)
        choice = np.random.choice(a = DataFrame.index, size = N, p = weight)
    if type == 'uniform':
        weight = [1/DataFrame.shape[0] for number in DataFrame.index]
        choice = np.random.choice(a = DataFrame.index, size = N, p = weight)
    
    # historic simulation
    if LastPrice == None:
        simulated_return = DataFrame.iloc[choice,:]
        return simulated_return
    else:
        simulated_return = DataFrame.iloc[choice,:]
        
        if DateOrNot == 'Y':
            simulated_return_wo_date = simulated_return.iloc[:,1:]
            
            if simulated_return_wo_date.shape[1] != len(LastPrice):
                raise ValueError("length of simulated return without date doesnt match the length of price")
            else:
                simulated_price = (simulated_return_wo_date + 1) * LastPrice
                simulated_PV = np.dot(simulated_price, Holding)
                simulated_PV = sorted(simulated_PV)
                
        elif DateOrNot == 'N':
            
            if simulated_return.shape[1] != len(LastPrice):
                raise ValueError("length of simulated return without date doesnt match the length of price")
            else:
                simulated_price = (simulated_return + 1) * LastPrice
                simulated_PV = np.dot(simulated_price, Holding)
                simulated_PV = sorted(simulated_PV)
                
        else:
            raise ValueError("DateOrNot must be 'Y' or 'N'.")
        
        return simulated_PV
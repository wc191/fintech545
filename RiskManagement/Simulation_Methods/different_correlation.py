import numpy as np

def exp_weighted_cov(input, lambda_=0.97):
    ror = input.values
    ror_mean = np.mean(ror, axis=0)
    dev = ror - ror_mean
    times = dev.shape[0]
    weights = np.zeros(times)
    
    for i in range(times):
        weights[times - i - 1]  = (1 - lambda_) * lambda_**i
    
    weights_mat = np.diag(weights/sum(weights))

    cov = np.transpose(dev) @ weights_mat @ dev
    return cov


def p_corr_var_f(returns):
    return np.cov(returns, rowvar=False)


def p_corr_ew_var_f(returns, lambda_=0.97):
    cov = exp_weighted_cov(returns, lambda_)
    std_dev = np.sqrt(np.diag(cov))
    corr = np.corrcoef(returns.T)
    cov = np.outer(std_dev, std_dev) * corr
    return cov


def ew_corr_p_var_f(returns, lambda_=0.97):
    cov = exp_weighted_cov(returns, lambda_)
        
    asset_std = np.diag(np.reciprocal(np.sqrt(np.diag(cov))))
    corr = asset_std @ cov @ asset_std.T
    
    var = np.var(returns)
    std_dev = np.sqrt(var)
    
    cov = np.outer(std_dev, std_dev) * corr
    return cov
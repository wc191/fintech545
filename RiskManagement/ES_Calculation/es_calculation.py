import numpy as np
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from ..Simulation_Methods import direct_simulation
from ..return_calculate import return_calculate

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

# VaR calculation methods (all discussed)
# Using a normal distribution.
def cal_VaR_ES_norm(returns, n=10000, alpha=0.05):
    mu = returns.mean()
    sigma = returns.std()
    simu_returns = np.random.normal(mu, sigma, n)
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns

# Using a normal distribution with an Exponentially Weighted variance
def cal_VaR_ES_ew_norm(returns, lambda_=0.94, n=10000, alpha=0.05):
    mu = returns.mean()
    sigma = np.sqrt(exp_weighted_cov(returns, lambda_=lambda_))
    simu_returns = np.random.normal(mu, sigma, n)
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns

# Using a MLE fitted T distribution.
def MLE_t(params, returns):
    df, loc, scale = params
    neg_LL = -1 * np.sum(stats.t.logpdf(returns, df=df, loc=loc, scale=scale))
    return(neg_LL)

def cal_VaR_ES_MLE_t(returns, n=10000, alpha=0.05):
    
    mu = returns.mean()[0]
    sigma = returns.std()[0]
    
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 1},
        {'type': 'ineq', 'fun': lambda x: x[2]}
    ]
    
    res = minimize(MLE_t, x0=[10, mu, sigma], args=(returns,), constraints=constraints)
    
    df, loc, scale = res.x
    simu_returns = stats.t.rvs(df, loc=loc, scale=scale, size=n)
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns

# Using a fitted AR(1) model.
def cal_VaR_ES_AR1(returns, n=10000, alpha=0.05):
    #a more general model that extends the ARMA model to non-stationary time series data.
    model = ARIMA(returns, order=(1, 0, 0)).fit()
    sigma = np.std(model.resid)
    simu_returns = np.empty(n)
    returns = returns.values
    for i in range(n):
        simu_returns[i] =  model.params[0] * (returns[-1]) + sigma * np.random.normal()
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns

# Using a Historic Simulation.
def cal_VaR_ES_hist(returns, alpha=0.05):
    returns = sorted(returns)
    n = alpha * returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (returns[iup] + returns[idn]) / 2

    ES = np.mean(returns[0:idn])

    return -VaR, -ES, returns

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
    simu_returns = np.random.normal(mu, sigma[0][0], n)
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
    n = alpha * returns.size
    returns = sorted(returns)
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (returns[iup] + returns[idn]) / 2

    ES = np.mean(returns[0:idn])

    return -VaR, -ES, returns

# Delta Normal VaR
def get_single_portfolio(portfolio, prices, portfolio_code):
    assets = portfolio[portfolio["Portfolio"] == portfolio_code]
        
    codes = list(assets["Stock"])
    assets_prices = prices[["Date"] + codes].copy()
    
    Total_Value = prices[codes].tail(1).values.dot(assets["Holding"].values)[0]
    holdings = assets["Holding"].values

    asset_values = holdings.reshape(-1, 1) * prices[codes].tail(1).T.values
    delta = asset_values / Total_Value
    
    return Total_Value, assets_prices, holdings, delta

def get_all_portfolio(portfolio, prices):
    assets = portfolio.drop('Portfolio',axis=1)
    assets = assets.groupby(["Stock"], as_index=False)["Holding"].sum()
        
    codes = list(assets["Stock"])
    assets_prices = prices[["Date"] + codes].copy()
    
    Total_Value = prices[codes].tail(1).values.dot(assets["Holding"].values)[0]
    holdings = assets["Holding"].values

    asset_values = holdings.reshape(-1, 1) * prices[codes].tail(1).T.values
    delta = asset_values / Total_Value
    
    return Total_Value, assets_prices, holdings, delta

def cal_delta_VaR(Total_Value, assets_prices, delta, alpha=0.05, lambda_=0.94):
    returns = return_calculate(assets_prices).drop('Date', axis=1)
    assets_cov = exp_weighted_cov(returns, lambda_)
    
    delta_norm_VaR = -Total_Value * stats.norm.ppf(alpha) * np.sqrt(delta.T @ assets_cov @ delta)
    
    return delta_norm_VaR.item()

#4.7 Monte Carlo VaR
def calculate_MC_var(assets_prices, holdings, alpha=0.05, lambda_=0.94, n_simulation = 10000):
    returns = return_calculate(assets_prices).drop('Date',axis=1)
    returns_norm = returns - returns.mean()
    assets_cov = exp_weighted_cov(returns_norm, lambda_)
    assets_prices = assets_prices.drop('Date',axis=1)
    np.random.seed(0)
    simu_returns = np.add(direct_simulation(assets_cov), returns.mean().values)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)
    MC_VaR = -np.percentile(simu_prices, alpha*100)
    return MC_VaR

# Historical VaR
def cal_hist_VaR(assets_prices, holdings, alpha=0.05):
    returns = return_calculate(assets_prices).drop("Date", axis=1)
    assets_prices = assets_prices.drop('Date',axis=1)
    simu_returns = returns.sample(1000, replace=True)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)

    hist_VaR = -np.percentile(simu_prices, alpha*100)

    return hist_VaR
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigvals
from scipy import stats
from scipy.optimize import minimize 
from statsmodels.tsa.arima.model import ARIMA

def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars = prices.columns
    nVars = len(vars)
    vars = [var for var in vars if var != dateColumn]
    if nVars == len(vars):
        raise ValueError("dateColumn: " + dateColumn + " not in DataFrame: " + str(vars))
    nVars = nVars - 1

    p = np.matrix(prices[vars])
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n-1,m))

    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError("method: " + method + " must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[dateColumn][1:n]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:,i]
    return out

# 1. Covariance estimation techniques.
# 1.1 exponentially weighted covariance matrix.
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

""" # 1.2 Use PCA to calculate the cumulative variance
def simulate_pca(a, nsim, nval=None):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eig(a)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    posv = np.where(vals >= 1e-8)[0]
    if nval is not None:
        if nval < len(posv):
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]

    cum_var = (np.cumsum(vals[:nsim]) / np.sum(vals))[-1]

    return cum_var """



# 2. Non PSD fixes for correlation matrices
# 2.1 Near PSD Matrix
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = np.matmul(np.matmul(invSD, out), invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.matmul(np.square(vecs), vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

# 2.2 Using Higham’s method to fix the matrix.
def Frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result

def Higham_psd(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        # PS
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        #PU
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = Frobenius(Yk-Y0)
        #norm = np.linalg.norm(Yk-Y0, ord='fro')
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk


def is_psd(matrix):
    vals = np.linalg.eigh(matrix)[0]
    return np.all(vals >= -1e-8)


# 3. Simulation Methods
#Cholesky that assumes PSD
def chol_psd(a):
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        temp = a[j, j] - s
        if temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            root[j, j+1:] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return root
# 3.1 #simulation directly from a covariance matrix or using PCA with an optional parameter for % variance explained.
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

# 3.2 Pearson correlation and var
def p_cov_var_f(returns):
    return np.cov(returns, rowvar=False)

# 3.3 Pearson correlation and EW variance
def p_corr_ew_var_f(returns, lambda_=0.97):
    cov = exp_weighted_cov(returns, lambda_)
    std_dev = np.sqrt(np.diag(cov))
    corr = np.corrcoef(returns.T)
    cov = np.outer(std_dev, std_dev) * corr
    return cov

# 3.4 EW(cov+corr) is exp_weighted_cov(input, lambda_)
#ew_corr_cov = exp_weighted_cov(df)

# 3.5 EW Corr +Var
def ew_corr_p_var_f(returns, lambda_=0.97):
    cov = exp_weighted_cov(returns, lambda_)
        
    asset_std = np.diag(np.reciprocal(np.sqrt(np.diag(cov))))
    corr = asset_std @ cov @ asset_std.T
    
    var = np.var(returns)
    std_dev = np.sqrt(var)
    
    cov = np.outer(std_dev, std_dev) * corr
    return cov



# 4. VaR calculation methods (all discussed)
# 4.1 Using a normal distribution.
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

# 4.2 Using a normal distribution with an Exponentially Weighted variance
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

# 4.3 Using a MLE fitted T distribution.
def MLE_t(params, returns):
    df, loc, scale = params
    neg_LL = -1 * np.sum(stats.t.logpdf(returns, df=df, loc=loc, scale=scale))
    return(neg_LL)

def cal_VaR_MLE_t(returns, n=10000, alpha=0.05):
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 1},
        {'type': 'ineq', 'fun': lambda x: x[2]}
    ]
    
    res = minimize(MLE_t, x0=[10, returns.mean(), returns.std()], args=(returns,), constraints=constraints)
    
    df, loc, scale = res.x
    simu_returns = stats.t.rvs(df, loc=loc, scale=scale, size=n)
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns

# 4.4 Using a fitted AR(1) model.
def cal_VaR_AR1(returns, n=10000, alpha=0.05):
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

# 4.5 Using a Historic Simulation.
def cal_VaR_hist(returns, alpha=0.05):
    n = alpha * returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (returns[iup] + returns[idn]) / 2

    ES = np.mean(returns[0:idn])

    return -VaR, -ES, returns


# 4.6 Delta Normal VaR
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
    simu_returns = np.add(multivar_norm_simu(assets_cov), returns.mean().values)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)
    MC_VaR = -np.percentile(simu_prices, alpha*100)
    return MC_VaR

# 4.8 Historical VaR
def cal_hist_VaR(assets_prices, holdings, alpha=0.05):
    returns = return_calculate(assets_prices).drop("Date", axis=1)
    assets_prices = assets_prices.drop('Date',axis=1)
    simu_returns = returns.sample(1000, replace=True)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)

    hist_VaR = -np.percentile(simu_prices, alpha*100)

    return hist_VaR
# import libraries
import pandas as pd
import os
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import normaltest
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
sb.set_theme()
import math
from scipy.optimize import minimize
from scipy.stats import t
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

########## problem 1 ##########

# compare the skewness/kurtosis in python package and skewness/kurtosis formula introduced in class

# first, we need to change directory
os.chdir(r'C:\Users\WANGLIN CAI\fintech545\Week02')

# generate the data from normal distribution
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0, 1, 1000)


# calculate the skewness using its original definition
def skew_cal(data):
    count = 0
    m = np.mean(data)
    for i in data:
        count += (i - m)**3/len(data)
    count = count/(np.std(data)**3)
    return count

print(skew_cal(data1))
print(skew_cal(data2))

print(skew(data1))
print(skew(data2))

# calculate the difference
skew_result1 = skew_cal(data1) - skew(data1)
skew_result2 = skew_cal(data2) - skew(data2)

print(skew_result1)
print(skew_result2)

# for the skewness
# we can see that the difference is very small, the reason why the result is not exactly 0 is probably because the rounding problem in the steps of calculations.


# calculate the kurtosis using its original definition
def kurtosis_cal(data):
    count = 0
    m = np.mean(data)
    for i in data:
        count += (i - m)**4/len(data)
    count = count/(np.std(data)**4)
    return count

print(kurtosis_cal(data1))
print(kurtosis_cal(data2))

print(kurtosis(data1))
print(kurtosis(data2))

# calculate the difference
kurtosis_result1 = kurtosis_cal(data1) - kurtosis(data1)
kurtosis_result2 = kurtosis_cal(data2) - kurtosis(data2)

print(kurtosis_result1)
print(kurtosis_result2)

# we can see the difference is 3 if we ignore the difference in decimal places.
# the package use excess kurtosis as kurtosis since the difference is 3
# again the reason why the result is not exactly 3 is probably because the rounding problem in the steps of calculations.


########## problem 2 ##########

# reading the dataset for problem2
dataset_p2 = pd.read_csv('problem2.csv')
# get first column equals to 1
dataset_p2.insert(loc=0,column='1',value=1)

############################ OLS method
# convert dataset into matrix
x_p2 = np.matrix(dataset_p2.iloc[:,0:2])
y_p2 = np.matrix(dataset_p2.iloc[:,2:3])

# calculation
beta_hat = np.linalg.inv(x_p2.T * x_p2) * x_p2.T * y_p2
y_predict = x_p2 * beta_hat
error_vector_ols = y_p2 - y_predict

# plot the histgram
hist_plot = sb.histplot(error_vector_ols, kde = True, stat="density")
# plot normal distribution
xmin, xmax = hist_plot.get_xlim()
normal_x = np.linspace(xmin, xmax, 100)
hist_plot.plot(normal_x, norm.pdf(normal_x), color = 'red')
# looks like it is not a normal distribution

# test if the error is normal distributed or not
normaltest(error_vector_ols)
# the results show that it is not from a normal distribution


############################# MLE Method

################# normal
def calcLogLikelihood_normal(guess, true, n):
    error = true-guess
    sigma = np.std(error)
    f = ((1/(2 * math.pi*sigma*sigma))**(n/2))* \
        np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)

def myFunction_normal(var):
    #   load my  data
    x = np.array(x_p2[:,1])
    y = np.array(y_p2)
    yGuess = (var[1]*x) + var[0]
    f = calcLogLikelihood_normal(yGuess, y, float(len(yGuess)))
    return (-1*f)

#  Let's pick some random starting points for the optimization    
nvar = 2
var = np.zeros(nvar)
var[0] = 0
var[1] = 1

#   let's maximize the likelihood (minimize -1*max(likelihood)
res_normal = minimize(myFunction_normal, var, method='BFGS',
                options={'disp': True})

error_vector_mle_normal = y_p2 - x_p2 * np.matrix(res_normal.x).T
# plot the histgram
plt.subplot(2, 2, 1)
hist_plot = sb.histplot(error_vector_mle_normal, kde = True, stat="density")
# plot normal distribution
xmin, xmax = hist_plot.get_xlim()
normal_x = np.linspace(xmin, xmax, 100)
hist_plot.plot(normal_x, norm.pdf(normal_x), color = 'red')
# looks like it is not a normal distribution

# test if the error is normal distributed or not
normaltest(error_vector_mle_normal)
# the results show that it is not from a normal distribution


################# t distribution
def calcLogLikelihood_t(guess, true, n):
    error = true-guess
    f = 1
    for i in error:
        f = f * t.pdf(i,n)
    return np.log(f)

def myFunction_t_2(var):
    #   load my  data
    x = np.array(x_p2[:,1])
    y = np.array(y_p2)
    yGuess = (var[1]*x) + var[0]
    f = calcLogLikelihood_t(yGuess, y, 2)
    return (-1*f)

def myFunction_t_10(var):
    #   load my  data
    x = np.array(x_p2[:,1])
    y = np.array(y_p2)
    yGuess = (var[1]*x) + var[0]
    f = calcLogLikelihood_t(yGuess, y, 10)
    return (-1*f)

#  Let's pick some random starting points for the optimization    
nvar = 2
var = np.zeros(nvar)
var[0] = 0
var[1] = 1

######################### t = 2
#   let's maximize the likelihood (minimize -1*max(likelihood)
res_t_2 = minimize(myFunction_t_2, var, method='BFGS',
                options={'disp': True})
res_t_2.x

error_vector_mle_t_2 = y_p2 - x_p2 * np.matrix(res_t_2.x).T
# plot the histgram
plt.subplot(2, 2, 2)
hist_plot = sb.histplot(error_vector_mle_t_2, kde = True, stat="density")
# plot normal distribution
xmin, xmax = hist_plot.get_xlim()
t_x = np.linspace(xmin, xmax, 100)
hist_plot.plot(t_x, t.pdf(normal_x,2), color = 'red')



######################## t = 10
#   let's maximize the likelihood (minimize -1*max(likelihood)
res_t_10 = minimize(myFunction_t_10, var, method='BFGS',
                options={'disp': True})
res_t_10.x

error_vector_mle_t_10 = y_p2 - x_p2 * np.matrix(res_t_10.x).T
# plot the histgram
plt.subplot(2, 2, 3)
hist_plot = sb.histplot(error_vector_mle_t_10, kde = True, stat="density")
# plot normal distribution
xmin, xmax = hist_plot.get_xlim()
t_x = np.linspace(xmin, xmax, 100)
hist_plot.plot(t_x, t.pdf(normal_x,10), color = 'red')


# goodness of fit
ss_total = 0
for i in y_p2:
    ss_total += (i[0,0]-np.mean(y_p2))**2

r_squared_normal = 1 - np.dot(error_vector_mle_normal.T, error_vector_mle_normal)[0,0]/ss_total
r_squared_t_2 = 1 - np.dot(error_vector_mle_t_2.T, error_vector_mle_t_2)[0,0]/ss_total
r_squared_t_10 = 1 - np.dot(error_vector_mle_t_10.T, error_vector_mle_t_10)[0,0]/ss_total

########## problem 3 ##########

# Generate some data for an AR(1) model
np.random.seed(1)
ar1_data = np.random.randn(100)
for i in range(1, 100):
    ar1_data[i] = 0.6 * ar1_data[i-1] + np.random.randn()

# Plot the ACF and PACF for AR(1)
plot_acf(ar1_data, lags=10)
plot_pacf(ar1_data, lags=10)

# Generate some data for an AR(2) model
np.random.seed(1)
ar2_data = np.random.randn(100)
for i in range(2, 100):
    ar2_data[i] = 0.6 * ar2_data[i-1] - 0.3 * ar2_data[i-2] + np.random.randn()

# Plot the ACF and PACF for AR(2)
plot_acf(ar2_data, lags=10)
plot_pacf(ar2_data, lags=10)

# Generate some data for an AR(3) model
np.random.seed(1)
ar3_data = np.random.randn(100)
for i in range(3, 100):
    ar3_data[i] = 0.6 * ar3_data[i-1] - 0.3 * ar3_data[i-2] + 0.2 * ar3_data[i-3] + np.random.randn()

# Plot the ACF and PACF for AR(3)
plot_acf(ar3_data, lags=10)
plot_pacf(ar3_data, lags=10)

# Generate some data for an MA(1) model
np.random.seed(1)
ma1_data = np.random.randn(100)
for i in range(1, 100):
    ma1_data[i] = 0.6 * np.random.randn() + ma1_data[i-1]

# Plot the ACF and PACF for MA(1)
plot_acf(ma1_data, lags=10)
plot_pacf(ma1_data, lags=10)

# Generate some data for an MA(2) model
np.random.seed(1)
ma2_data = np.random.randn(100)
for i in range(2, 100):
    ma2_data[i] = 0.6 * np.random.randn() + 0.3 * np.random.randn() + ma2_data[i-1] - 0.3 * ma2_data[i-2]

# Plot the ACF and PACF for MA(2)
plot_acf(ma2_data, lags=10)
plot_pacf(ma2_data, lags=10)

# Generate some data for an MA(3) model
np.random.seed(1)
ma3_data = np.random.randn(100)
for i in range(3, 100):
    ma3_data[i] = 0.6 * np.random.randn() + 0.3 * np.random.randn() + 0.2 * np.random.randn() + ma3_data[i-1] - 0.3 * ma3_data[i-2] + 0.2 * ma3_data[i-3]

# Plot the ACF and PACF for MA(3)
plot_acf(ma3_data, lags=10)
plot_pacf(ma3_data, lags=10)



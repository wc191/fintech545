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

########## problem 1 ##########

# compare the skewness/kurtosis in python package and skewness/kurtosis formula introduced in class

# first, we need to change directory
os.chdir(r'C:\Users\WANGLIN CAI\fintech545\Week02')

# generate the data from normal distribution
data1 = np.random.normal(0, 1, 100)
data1 = np.random.normal(0, 1, 1000)


# calculate the skewness using its original definition
def skew_cal(data):
    count = 0
    m = np.mean(data)
    for i in data:
        count += (i - m)**3/len(data)
    count = count/(np.std(data)**3)
    return count

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

# OLS method
# convert dataset into matrix
x_p2 = np.matrix(dataset_p2.iloc[:,0:2])
y_p2 = np.matrix(dataset_p2.iloc[:,2:3])

# calculation
beta_hat = np.linalg.inv(x_p2.T * x_p2) * x_p2.T * y_p2
y_predict = x_p2 * beta_hat
error_vector = y_p2 - y_predict

# plot the histgram
hist_plot = sb.histplot(error_vector, kde = True, stat="density")
# plot normal distribution
xmin, xmax = hist_plot.get_xlim()
normal_x = np.linspace(xmin, xmax, 100)
hist_plot.plot(normal_x, norm.pdf(normal_x), color = 'red')
# looks like it is not a normal distribution

# test if the error is normal distributed or not
normaltest(error_vector)
# the results show that it is not from a normal distribution

########## problem 3 ##########




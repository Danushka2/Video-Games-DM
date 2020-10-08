# -- coding: utf-8 --

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from dataset.cleanDs import cleanDs

from sklearn.model_selection import train_test_split

cleanData = cleanDs()
df = cleanData.clean_db()

x = df['Global_Sales']
y = df['User_Score']

# plt.xlim(0,20)
# plt.ylim(0,12)

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y, color='green')
plt.title('Global_Sales Vs User_Score', fontsize=14)
plt.xlabel('Global_Sales', fontsize=14)
plt.ylabel('User_Score', fontsize=14)
plt.grid(True)
plt.plot(x, mymodel)
plt.show()
import numpy

print(numpy.corrcoef(x, y))

x = df['Critic_Score']
y = df['User_Score']

# plt.xlim(0,20)
# plt.ylim(0,12)

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y, color='green')
plt.title('Critic_Score Vs User_Score', fontsize=14)
plt.xlabel('Critic_Score', fontsize=14)
plt.ylabel('User_Score', fontsize=14)
plt.grid(True)
plt.plot(x, mymodel)
plt.show()
import numpy

print(numpy.corrcoef(x, y))


x = df['Critic_Count']
y = df['User_Score']

#plt.xlim(0,5)
#plt.ylim(0,1000)

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))
plt.scatter(x,y , color='green')
plt.title('Critic_Count Vs User_Score', fontsize=14)
plt.xlabel('Critic_Count', fontsize=14)
plt.ylabel('User_Score', fontsize=14)
plt.grid(True)
plt.plot(x, mymodel)
plt.show()
import numpy
print(numpy.corrcoef(x,y))

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

# df = pd.df(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

X = df[['Global_Sales', 'Critic_Count',
        'Critic_Score']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['User_Score']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
Global_Sales = 23.21
Critic_Score = 91
Critic_Count = 64

print('Predicted User Score: \n', regr.predict([[Global_Sales, Critic_Score, Critic_Count]]))

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)
# -- coding: utf-8 --

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from dataset.cleanDs import cleanDs
from sklearn.model_selection import train_test_split

cleanData = cleanDs()
df = cleanData.clean_db()

# Creating plot with Global_Sales & User_Score
x = df['Global_Sales']
y = df['User_Score']

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

print("Coefficient between Global_Sales Vs User_Score",numpy.corrcoef(x, y))

# Creating plot with Critic_Score & User_Score
x = df['Critic_Score']
y = df['User_Score']

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

print("Coefficient between Critic_Score Vs User_Score", numpy.corrcoef(x, y))

# Creating plot with Critic_Count & User_Score
x = df['Critic_Count']
y = df['User_Score']

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y, color='green')
plt.title('Critic_Count Vs User_Score', fontsize=14)
plt.xlabel('Critic_Count', fontsize=14)
plt.ylabel('User_Score', fontsize=14)
plt.grid(True)
plt.plot(x, mymodel)
plt.show()
import numpy

print("Coefficient between Critic_Count Vs User_Score", numpy.corrcoef(x, y))



import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

X = df[['Critic_Score','Global_Sales','Critic_Count']]
Y = df['User_Score']

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
y_pred = regr.predict(X_test)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

print(pd.DataFrame({'Actual':Y_test,'Predicted':y_pred}))

# prediction with sklearn
Global_Sales = 82.53
Critic_Score = 76
Critic_Count = 51

print('Predicted User Score: \n', regr.predict([[Global_Sales, Critic_Score, Critic_Count]]))

# with statsmodels
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

# To find Accuracy
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test,y_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test,y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(Y_test,y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(Y_test,y_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test,y_pred), 2))

print(regr.score(X_test,Y_test))

print("-------------------------------------------")

# Create a serializable object with pickle
# saving model to disk
pickle.dump(regr,open('MultipleLinearRegression.pkl','wb'))

# testing the model
model = pickle.load(open('MultipleLinearRegression.pkl','rb'))
# predict using a value


df[['Critic_Score','Global_Sales','Critic_Count']].iloc[0]

new_row = {'Critic_Score':76, 'Global_Sales':82.53, 'Critic_Count':51}
df_Apnd = df.append(new_row, ignore_index=True)

df_Apnd[['Critic_Score','Global_Sales','Critic_Count']].iloc[-1]

# Get features and labels
newDF2 = df_Apnd[['Critic_Score','Global_Sales','Critic_Count']]
M = newDF2
N = df_Apnd.iloc[:,3].values

# Encoding
objList2 = M.select_dtypes(include = "object").columns


l = M.iloc[-1].values

print(model.predict(l.reshape(1,-1)))

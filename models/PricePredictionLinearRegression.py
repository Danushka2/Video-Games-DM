
# Multiple Linear Regression

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

#import dataset
dataset = pd.read_csv('dataset/Video_Games_Sales_as_at_22_Dec_2016.csv')
XY_Combined = dataset[['Name','Year_of_Release','Platform','Genre','Publisher','Developer','Global_Sales']]

# remove missing values
XY_DropNA = XY_Combined.dropna()

# Extract X and Y
Y = XY_DropNA['Global_Sales'].values
X = XY_DropNA[['Name','Year_of_Release','Platform','Genre','Publisher','Developer']]

# Get copy of X and extract strings
x = X.copy()
objList = x.select_dtypes(include = "object").columns

# encode the strings to numeric values
le = LabelEncoder()

for feat in objList:
    x[feat] = le.fit_transform(x[feat].astype(str))

# convert the year_of_release to int
x['Year_of_Release'] = x['Year_of_Release'].astype(int)

# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.5, random_state = 42)

# Do feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
# feature scale x train and x test
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
# feature scale y train
y_train = sc_Y.fit_transform(y_train.reshape(-1,1))


#Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Implement backward elimination with significance of 5%
import statsmodels.api as sm

# add b0 constant
x = np.append(arr=np.ones((9904,1)).astype(int), values=x ,axis=1)

#x_opt = x[:,[0,1,2,3,4,5,6]]
#regressor_ols = sm.OLS(endog = Y,exog = x_opt).fit()
#regressor_ols.summary()

# final regression model only depend on developer ,platform and year
x_opt = x[:,[0,2,3,6]]
regressor_ols = sm.OLS(endog = Y,exog = x_opt).fit()
regressor_ols.summary()

# Create a serializable object with pickle
# saving model to disk
pickle.dump(regressor_ols,open('regressionModel.pkl','wb'))

# testing the model
model = pickle.load(open('regressionModel.pkl','rb'))

le = LabelEncoder()
k = le.fit_transform([1,'PC',2021,'Midori'])

print(model.predict(k))


















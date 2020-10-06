# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:13:31 2020

@author: sudil
"""

# multiple linear regression
from dataset.cleanDs import cleanDs
from sklearn.preprocessing import LabelEncoder
import pickle

################## get data from cleanDs    ############################
cleanData = cleanDs()
df = cleanData.clean_db()


##################  Get features and labels     ######################
X = df[['Year_of_Release','Platform','Genre','Publisher','Developer']]
#X['Year_of_Release'] = X['Year_of_Release'].astype(int)
#X = X.iloc[:].values
Y = df.iloc[:,5].values

###################  Encoding #########################
# 01 One hot encoding 
# encode categorical values
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer

# enter the column which is going to onehotencode
ct = ColumnTransformer([("Platform", OneHotEncoder(), [1]),
                        ("Genre", OneHotEncoder(), [2]),
                        ("Publisher", OneHotEncoder(), [3]),
                        ("Developer", OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X).toarray()

# 02 Convert to numeric labels
# encode X with arbitary values
# encode the strings to numeric values
objList = X.select_dtypes(include = "object").columns
# encode the strings to numeric values
le = LabelEncoder()

for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))
#X = onehotencoder.fit_transform(X).toarray()

#############   Linear regression    ##################################

# Python linear regression model take care of dummy variable trap so
# need not to remove dummy variables

# split to training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 42)

# fitting multiple linear regression in training model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train) #0.3445800562295963 with one hot encode

# prediction of test set results
y_pred = regressor.predict(y_test)

# select the optimal model
# Implement backward elimination with significance of 5%
#import statsmodels.api as sm

# add b0 constant
#X = np.append(arr=np.ones((16416,1)).astype(int), values=X ,axis=1)

#x_opt = X[:]
#regressor_ols = sm.OLS(endog = Y,exog = x_opt).fit()
#regressor_ols.summary()
#regressor.score(X_train,y_train) #0.3445800562295963
#xn = regressor.coef_
#xn[0]

################    Polynomial Linear Regression ############################

# MemoryError: Unable to allocate 325. GiB for an array with shape (16416, 2657665) and data type float64
# use polynormial regression # no point of doing ;)
# https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
from sklearn.preprocessing import PolynomialFeatures
regressor_polynorm = PolynomialFeatures(degree=2)
x_poly = regressor_polynorm.fit_transform(X)

#create training and testing
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly,Y, test_size=0.2, random_state = 42)

regressor_polynorm.fit(X_train_poly,y_train_poly)
regressor_polynorm.score

#########################   SVR   ######################################
# need to do feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
#X_feaScale = sc_X.fit_transform(X) # with one hot encodin
X_feaScale = sc_X.fit_transform(X.iloc[:].values) # with numeric labels
Y_feaScale = sc_y.fit_transform(Y.reshape(-1,1))

from sklearn.svm import SVR
# pass kernel as parameter
svr = SVR(kernel='rbf')
svr.fit(X_feaScale,Y_feaScale.ravel())
svr.score(X_feaScale,Y_feaScale) #0.1745800562295963 with labelEn 0.00445846999863464

######################  Decision Tree Regression model #####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state = 42)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=42)
#dtr.fit(X,Y)
#dtr.score(X,Y) # 8774636801425665

# with label encoding
dtr.fit(X_train,y_train.reshape(-1,1))
dtr.score(X_train,y_train.reshape(-1,1)) # 0.8774636801425665
y_pred = dtr.predict(X_test)

# Create a serializable object with pickle
# saving model to disk
pickle.dump(dtr,open('DecisionTreeRegression.pkl','wb'))

# testing the model
model = pickle.load(open('DecisionTreeRegression.pkl','rb'))
# predict using a value
k = le.fit_transform([2028,'PS2','Strategy','EA','NFS'])

print(model.predict(k.reshape(1,-1)))


####################    Random forest regression  #########################
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(X,Y)
rfr.score(X,Y) # 0.824814361730428

##################      ANN  ##############################################
# need to do feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_feaScale = sc_X.fit_transform(X.iloc[:].values) # with numeric labels


import keras
# sequential model to initialize NN
from keras.models import Sequential
# Dense model for build layers
from keras.layers import Dense

# initailize ANN
ann_regressor = Sequential()

# Adding input layers #output_dim = input_dim+1/2 not a rule
ann_regressor.add(Dense(5,kernel_initializer='normal',activation='linear',input_dim = 5))

# Adding hidden layer 
ann_regressor.add(Dense(20,kernel_initializer='normal',activation='linear'))
ann_regressor.add(Dense(10,kernel_initializer='normal',activation='linear'))

# Adding output layer
ann_regressor.add(Dense(1,kernel_initializer='normal',activation='linear'))

# run the model
ann_regressor.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

#  connect ann
ann_regressor.fit(X,Y,3,100)

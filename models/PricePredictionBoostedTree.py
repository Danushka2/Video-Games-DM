# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:45:54 2020

@author: sudil
"""

from dataset.cleanDs import cleanDs
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# get data from cleanDs
cleanData = cleanDs()
df = cleanData.clean_db()

X = df[['Year_of_Release','Platform','Genre','Publisher','Developer']]
Y = df['Global_Sales'].values
X['Year_of_Release'] = X['Year_of_Release'].astype(str)

# encode the X values
#char_cols = X.dtypes.pipe(lambda x: x[x == 'object']).index

#for c in char_cols:
#    X[c] = pd.factorize(X[c])[0]

#print(X.nunique(dropna=False))

# do feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#x_trans = sc_X.fit_transform(X)

# encode x binary
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_results_0 = lb.fit_transform(X['Platform'])
lb_results_df_0 = pd.DataFrame(lb_results_0, columns=lb.classes_)
lb_results_df_0.drop(lb_results_df_0.columns[len(lb_results_df_0.columns)-1], axis=1, inplace=True)

lb_results_1 = lb.fit_transform(X['Genre'])
lb_results_df_1 = pd.DataFrame(lb_results_1, columns=lb.classes_)
lb_results_df_1.drop(lb_results_df_1.columns[len(lb_results_df_1.columns)-1], axis=1, inplace=True)

lb_results_2 = lb.fit_transform(X['Publisher'])
lb_results_df_2 = pd.DataFrame(lb_results_2, columns=lb.classes_)
lb_results_df_2.drop(lb_results_df_2.columns[len(lb_results_df_2.columns)-1], axis=1, inplace=True)

lb_results_3 = lb.fit_transform(X['Developer'])
lb_results_df_3 = pd.DataFrame(lb_results_3, columns=lb.classes_)
lb_results_df_3.drop(lb_results_df_3.columns[len(lb_results_df_3.columns)-1], axis=1, inplace=True)

result_df = pd.concat([lb_results_df_0, lb_results_df_1,lb_results_df_2,lb_results_df_3], axis=1)
result_df['Year_of_Release'] = X['Year_of_Release']

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.5, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(result_df,Y, test_size=0.6, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(x_trans,Y, test_size=0.5, random_state = 42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train,y_train) #0.05 #0.35
regressor.coef_

# use gradient boost classifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [1, 3, 5, 7, 9],
    'learning_rate': [0.01,0.1,1,10,100]
}

cv = GridSearchCV(gb,parameters,cv=5)
cv.fit(X_train,y_train.values.ravel())
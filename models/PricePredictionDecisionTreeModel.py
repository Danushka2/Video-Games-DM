# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:14:57 2020

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
newDF = df[['Year_of_Release','Platform','Genre','Publisher','Developer']]
X = newDF
#X['Year_of_Release'] = X['Year_of_Release'].astype(int)
#X = X.iloc[:].values
Y = df.iloc[:,5].values

###################  Encoding #########################
# 02 Convert to numeric labels
# encode X with arbitary values
# encode the strings to numeric values
objList = X.select_dtypes(include = "object").columns
# encode the strings to numeric values
le = LabelEncoder()

for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))
    
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
#k = le.fit_transform([2028,'PS2','Strategy','EA','NFS'])
#k = le.fit_transform([1985,'NES','Platform','Nintendo','Open Source'])


df[['Year_of_Release','Platform','Genre','Publisher','Developer']].iloc[0]

new_row = {'Year_of_Release':1985, 'Platform':'NES', 'Genre':'Platform', 'Publisher':'Nintendo', 'Developer':'Open Source'}
df_Apnd = df.append(new_row, ignore_index=True)

df_Apnd[['Year_of_Release','Platform','Genre','Publisher','Developer']].iloc[-1]

##################  Get features and labels     ######################
newDF2 = df_Apnd[['Year_of_Release','Platform','Genre','Publisher','Developer']]
M = newDF2
N = df_Apnd.iloc[:,5].values

###################  Encoding #########################
objList2 = M.select_dtypes(include = "object").columns
le = LabelEncoder()

for feat2 in objList2:
    M[feat2] = le.fit_transform(M[feat2].astype(str))


l = M.iloc[-1].values

#print(model.predict(k.reshape(1,-1)))
print(model.predict(l.reshape(1,-1)))



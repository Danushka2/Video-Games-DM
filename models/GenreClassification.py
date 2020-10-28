# -*- coding: utf-8 -*-

from dataset.cleanDs import cleanDs
import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle


#cleanedDS = dataset.clean_db()
cleanedDS = cleanDs()
df=cleanedDS.clean_db()



newDF = df[['Name','Platform','Year_of_Release','Publisher','Rating']]
X = newDF

Y = df.iloc[:,3].values

objList = ['Name','Platform','Year_of_Release','Publisher','Rating']

le = LabelEncoder()

for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))
    
    
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=46)

# Building and evaluating  classification Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg ="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
    
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)

pickle.dump(KNN,open('GenreClassifier.pkl','wb'))


# testing the model
model = pickle.load(open('GenreClassifier.pkl','rb'))

new_row = {'Name': 'Call of Duty: Black Ops II','Platform':'PS3','Year_of_Release':2012, 'Publisher':'Activision', 'Rating':'E'}
df_Apnd = df.append(new_row, ignore_index=True)

newDF2 = df_Apnd[['Name','Platform','Year_of_Release','Publisher','Rating']]
M = newDF2

objList2 = ['Name','Platform','Year_of_Release','Publisher','Rating']
le = LabelEncoder()

for feat2 in objList2:
    M[feat2] = le.fit_transform(M[feat2].astype(str))

l = M.iloc[-1].values

pred = model.predict(l.reshape(-1, 5))

print(pred)
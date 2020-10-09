# -- coding: utf-8 --

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset.cleanDs import cleanDs
from sklearn.preprocessing import LabelEncoder


# Get data from cleanDs
cleanData = cleanDs()
df = cleanData.clean_db()

# Get features and labels
newDFPC = df[['User_Score','Global_Sales','Critic_Score']]
X = newDFPC

# Encoding
# encode X with arbitary values
# encode the strings to numeric values
objList = X.select_dtypes(include = "object").columns

# encode the strings to numeric values
le = LabelEncoder()

for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))


# Find Optimal Clusters
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# within cluster sum of squared
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("number of clusters")
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4,init='k-means++',random_state=42)
y = kmeans.fit_predict(X)

# Visualising the clusters
# Creating figure
fig = plt.figure(figsize = (10, 7))
grp = plt.axes(projection ="3d")

grp.scatter3D(X.values[y == 0, 0], X.values[y == 0, 1],X.values[y == 0, 2], s=100, c = 'yellow', label = 'A')
grp.scatter3D(X.values[y == 1, 0], X.values[y == 1, 1],X.values[y == 1, 2], s=100, c = 'cyan', label = 'B')
grp.scatter3D(X.values[y == 2, 0], X.values[y == 2, 1],X.values[y == 2, 2], s=100, c = 'pink', label = 'C')
grp.scatter3D(X.values[y == 3, 0], X.values[y == 3, 1],X.values[y == 3, 2], s=100, c = 'plum', label = 'C')

grp.scatter3D(kmeans.cluster_centers_[:,0],
             kmeans.cluster_centers_[:,1],
             kmeans.cluster_centers_[:,2],
             s = 200,
             c = 'black', label = 'centroid')

plt.title('Clusters of games and ratings')
grp.set_xlabel('User_Score')
grp.set_ylabel('Global_Sales')
grp.set_zlabel('Critic_Score')
plt.legend()
plt.show()


# Create a serializable object with pickle
# saving model to disk
pickle.dump(kmeans,open('ClusteringPredict.pkl','wb'))

# testing the model
model = pickle.load(open('ClusteringPredict.pkl','rb'))


df[['User_Score','Global_Sales','Critic_Score']].iloc[0]

new_row = {'User_Score':1, 'Global_Sales':12, 'Critic_Score':20}
df_Apnd = df.append(new_row, ignore_index=True)

df_Apnd[['User_Score','Global_Sales','Critic_Score']].iloc[-1]

# Get features and labels
newDF2 = df_Apnd[['User_Score','Global_Sales','Critic_Score']]
M = newDF2
N = df_Apnd.iloc[:,3].values

# Encoding
objList2 = M.select_dtypes(include = "object").columns
le = LabelEncoder()

for feat2 in objList2:
    M[feat2] = le.fit_transform(M[feat2].astype(str))


l = M.iloc[-1].values

# print(model.predict(k.reshape(1,-1)))
clstr = model.predict(l.reshape(1,-1))
if clstr == 0:
    print("Cluster A")
elif clstr == 1:
    print("Cluster B")
elif clstr == 2:
    print("Cluster C")
else:
    print("Cluster D")
print(clstr)




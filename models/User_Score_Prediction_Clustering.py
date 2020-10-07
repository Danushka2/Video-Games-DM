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
X = df[['User_Score','Platform','Critic_Score']]


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
grp.scatter3D(X.values[y == 1, 0], X.values[y == 1, 1],X.values[y == 1, 2], s=100, c = 'blue', label = 'B')
grp.scatter3D(X.values[y == 2, 0], X.values[y == 2, 1],X.values[y == 2, 2], s=100, c = 'green', label = 'C')
grp.scatter3D(X.values[y == 3, 0], X.values[y == 3, 1],X.values[y == 3, 2], s=100, c = 'cyan', label = 'C')

grp.scatter3D(kmeans.cluster_centers_[:,0],
             kmeans.cluster_centers_[:,1],
             kmeans.cluster_centers_[:,2],
             s = 300,
             c = 'red', label = 'centroid')

plt.title('Clusters of games and ratings')
grp.set_xlabel('User_Score')
grp.set_ylabel('Platform')
grp.set_zlabel('Critic_Score')
plt.legend()
plt.show()

# Predicted Values shown in below
print(kmeans.predict((np.array([8.5,26,89])).reshape(1,-1)))


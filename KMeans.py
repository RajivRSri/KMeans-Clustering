#Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Set Working Directory
os.chdir("C:/R")

#Read Dataset
data = pd.read_csv("Mall_Customers.csv")

#Extract "Income" and "Score" variables
X = data.iloc[:,[3,4]].values

#Elbow method to find optimum number of clusters
from sklearn.cluster import KMeans

squareSum = []
for index in range (1, 10):
   centroid = KMeans(n_clusters = index,
                  init = "k-means++",
                  random_state = 2)
   centroid.fit(X)
   squareSum.append(centroid.inertia_)
   
plt.plot(range(1, 10), squareSum)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Build Clusters
kmeans = KMeans(n_clusters = 5,
                init = "k-means++",
                random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Cluster visualization
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "yellow", label = "Cluster1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "green", label = "Cluster2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "blue", label = "Cluster3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = "red", label = "Centroids")
plt.title("Cluster of Customers")
plt.xlabel("Annual Income($)")
plt.ylabel("Spending Power (1 - 100)")
plt.legend()
plt.show()








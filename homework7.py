from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)


# Set number of clusters
k = 3

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)


data['kmeans_cluster'] = kmeans.labels_

# Plot the clusters (using Sepal Length vs. Petal Length as an example)
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['kmeans_cluster'], cmap='viridis', marker='o')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title("K-Means Clustering on Iris Dataset")
plt.colorbar(label='Cluster')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


linkage_method = 'ward'
Z = linkage(data.iloc[:, :4], method=linkage_method)

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()


hierarchical = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
data['hierarchical_cluster'] = hierarchical.fit_predict(data.iloc[:, :4])

# Plot clusters for hierarchical clustering
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['hierarchical_cluster'], cmap='viridis', marker='o')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title("Hierarchical Clustering on Iris Dataset")
plt.colorbar(label='Cluster')
plt.show()

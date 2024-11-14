from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
labels = iris.target  

plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=labels, cmap='viridis')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Data - Sepal Length vs Petal Length")
plt.colorbar(label='True Label')
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(data_scaled)

# Predict clusters
gmm_clusters = gmm.predict(data_scaled)

# Plot GMM clustering results
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=gmm_clusters, cmap='viridis')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("GMM Clustering on Iris Dataset")
plt.colorbar(label='Cluster')
plt.show()



accuracy = accuracy_score(labels, gmm_clusters)
ari = adjusted_rand_score(labels, gmm_clusters)

print("Accuracy:", accuracy)
print("Adjusted Rand Index (ARI):", ari)

bic_values = []
aic_values = []
components_range = range(1, 7)

for n in components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(data_scaled)
    bic_values.append(gmm.bic(data_scaled))
    aic_values.append(gmm.aic(data_scaled))


plt.plot(components_range, bic_values, label='BIC', marker='o')
plt.plot(components_range, aic_values, label='AIC', marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.title("BIC and AIC for GMM")
plt.legend()
plt.show()

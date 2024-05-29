import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Data Collection and Cleaning
df = pd.read_csv('retractions35215.csv', lineterminator='\n')

# Data Cleaning
df.dropna(subset=['Country', 'Publisher'], inplace=True)

# Step 2: Encode the Categorical Variables
# One-hot encode 'Country'
onehot = OneHotEncoder()
country_encoded = onehot.fit_transform(df[['Country']]).toarray()

# Label encode 'Publisher'
label_encoder = LabelEncoder()
publisher_encoded = label_encoder.fit_transform(df['Publisher']).reshape(-1, 1)

# Combine the encoded features
X_encoded = np.hstack((country_encoded, publisher_encoded))

# Function to perform clustering and evaluation
def perform_clustering(X, standardize=False):
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # Evaluate
    silhouette = silhouette_score(X_pca, clusters)
    inertia = kmeans.inertia_

    return X_pca, clusters, silhouette, inertia

# Clustering without standardization
X_pca_no_scaling, clusters_no_scaling, silhouette_no_scaling, inertia_no_scaling = perform_clustering(X_encoded, standardize=False)

# Clustering with standardization
X_pca_with_scaling, clusters_with_scaling, silhouette_with_scaling, inertia_with_scaling = perform_clustering(X_encoded, standardize=True)

# Visualization Function
def plot_clusters(X_pca, clusters, title):
    plt.figure(figsize=(10, 7))
    for i in range(3):
        plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

# Visualization
plot_clusters(X_pca_no_scaling, clusters_no_scaling, 'Clustering without Standardization')
plot_clusters(X_pca_with_scaling, clusters_with_scaling, 'Clustering with Standardization')

# Step 5: Comparison
print(f'Silhouette Score Comparison:')
print(f'Without Standardization: {silhouette_no_scaling}')
print(f'With Standardization: {silhouette_with_scaling}')

print(f'\nInertia Comparison:')
print(f'Without Standardization: {inertia_no_scaling}')
print(f'With Standardization: {inertia_with_scaling}')

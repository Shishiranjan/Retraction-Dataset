import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
#import seaborn as sns

# Load the data
df = pd.read_csv('retractions35215.csv', lineterminator='\n')

# Data Cleaning
df.dropna(subset=['CitationCount', 'Journal', 'Publisher', 'Author', 'ArticleType', 'Country', 'RetractionNature'], inplace=True)
df['RetractionNature'] = np.where(df['RetractionNature'] == 'Retraction', 1, 0)

# Encode categorical variables
onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

# One-hot encode 'Country'
country_encoded = onehot_encoder.fit_transform(df[['Country']]).toarray()

# Label encode 'Publisher'
publisher_encoded = label_encoder.fit_transform(df['Publisher'])

# Combine encoded features into a single dataframe
X = np.hstack((country_encoded, publisher_encoded.reshape(-1, 1)))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to perform clustering and evaluate results
def cluster_and_evaluate(X, n_clusters=5, standardize=False):
    if standardize:
        X = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Agglomerative clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_labels = agglo.fit_predict(X)
    
    # Evaluate clustering performance
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)
    kmeans_calinski_harabasz = calinski_harabasz_score(X, kmeans_labels)
    
    agglo_silhouette = silhouette_score(X, agglo_labels)
    agglo_davies_bouldin = davies_bouldin_score(X, agglo_labels)
    agglo_calinski_harabasz = calinski_harabasz_score(X, agglo_labels)
    
    # Handle DBSCAN evaluation separately due to possible outliers
    if len(set(dbscan_labels)) > 1:
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
        dbscan_davies_bouldin = davies_bouldin_score(X, dbscan_labels)
        dbscan_calinski_harabasz = calinski_harabasz_score(X, dbscan_labels)
    else:
        dbscan_silhouette = -1
        dbscan_davies_bouldin = -1
        dbscan_calinski_harabasz = -1
    
    return {
        'kmeans': (kmeans_silhouette, kmeans_davies_bouldin, kmeans_calinski_harabasz),
        'dbscan': (dbscan_silhouette, dbscan_davies_bouldin, dbscan_calinski_harabasz),
        'agglo': (agglo_silhouette, agglo_davies_bouldin, agglo_calinski_harabasz)
    }

# Evaluate clustering without standardization
results_without_standardization = cluster_and_evaluate(X, n_clusters=5, standardize=False)

# Evaluate clustering with standardization
results_with_standardization = cluster_and_evaluate(X, n_clusters=5, standardize=True)

# Display results
print("Results without Standardization:")
print(results_without_standardization)
print("\nResults with Standardization:")
print(results_with_standardization)

# Visualize clustering results
def plot_clusters(X, labels, title):
    plt.figure(figsize=(10, 7))
    #sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=50)
    plt.title(title)
    plt.show()

# Visualize KMeans clustering results
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
plot_clusters(X, kmeans_labels, 'KMeans Clustering (without Standardization)')

kmeans_labels_standardized = kmeans.fit_predict(X_scaled)
plot_clusters(X_scaled, kmeans_labels_standardized, 'KMeans Clustering (with Standardization)')

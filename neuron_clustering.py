from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from combined_correlation_creator import get_united_data_as_df

def elbow_method(data, max_clusters=10):
    """
    Use the Elbow Method to find the optimal number of clusters.
    """
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plot the inertia values for different number of clusters
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertia, 'bo-', markersize=8)
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.grid(True)
    plt.show()

def silhouette_method(data, max_clusters=10):
    """
    Use the Silhouette Score to find the optimal number of clusters.
    """
    silhouette_avg = []
    for k in range(2, max_clusters + 1):  # Silhouette score is not defined for k=1
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_avg.append(score)

    # Plot the silhouette scores for different number of clusters
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_avg, 'bo-', markersize=8)
    plt.title('Silhouette Method for Optimal Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

def get_optimal_clusters(data, max_clusters=10):
    elbow_method(data, max_clusters)
    silhouette_method(data, max_clusters)

def preprocess_data(df, n_neighbors=5):
    """
    Preprocess the data by scaling the features.
    Standard scaling ensures that each feature has a mean of 0 and standard deviation of 1.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    print('Preprocessed data...')
    return scaled_data

def reduce_dimensionality(data, n_components):
    """
    Apply PCA to reduce dimensionality of the data for visualization purposes.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', markersize=8)
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.7, color='r', linestyle='--', label=f'Threshold')
    plt.xticks(range(1, len(cumulative_variance) + 1, max(1, len(cumulative_variance)//10)))  # Control the number of ticks on x-axis
    plt.grid(True)
    plt.legend()
    # plt.show()
    ## print the explained variance ratio
    print('Reduced data...')
    return reduced_data

def perform_clustering(neuron_names, data, method='kmeans', n_clusters=3):
    """
    Perform clustering on the neuron activity data.
    Supported methods: 'kmeans', 'dbscan'
    """
    if method == 'kmeans':        
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        clustering_model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Unsupported clustering method")

    labels = clustering_model.fit_predict(data)
    cluster_list = get_cluster_list_of_lists(labels, neuron_names)
    print('Performed clustering...')
    return cluster_list

def plot_clusters(reduced_data, labels):
    """
    Plot the clusters after dimensionality reduction.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', s=100)
    plt.title('Neuron Activity Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

def get_cluster_list_of_lists(labels, neuron_names):
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(neuron_names[idx])  # Append neuron names instead of indices
    
    # Convert the clusters dictionary into a list of lists
    cluster_list = list(clusters.values())
    return cluster_list

def main():
    activity_data = get_united_data_as_df().T
    ## Get neuron names from the rows of the dataframe
    neuron_names = list(activity_data.index)
    scaled_data = preprocess_data(activity_data)
    reduced_data = reduce_dimensionality(scaled_data, n_components=5)
    get_optimal_clusters(reduced_data, max_clusters=10)
    clusters_list = perform_clustering(neuron_names, reduced_data, method='kmeans', n_clusters=4)
    for cluster, idx in zip(clusters_list, range(len(clusters_list))):
        print(f'\nCluster {idx}: {cluster}. \nSize: {len(cluster)}')
    # plot_clusters(reduced_data, labels)
    return clusters_list

def get_clustering():
    activity_data = get_united_data_as_df().T
    neuron_names = list(activity_data.index)
    scaled_data = preprocess_data(activity_data)
    reduced_data = reduce_dimensionality(scaled_data, n_components=70)
    clusters_list = perform_clustering(neuron_names, reduced_data, method='kmeans', n_clusters=4)
    return clusters_list

if __name__ == '__main__':
    main()

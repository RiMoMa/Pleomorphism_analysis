import umap
import hdbscan
import matplotlib.pyplot as plt

def cluster_embeddings(embeddings, algorithm="HDBSCAN"):
    """Aplica clustering a los embeddings."""
    reducer = umap.UMAP(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    if algorithm == "HDBSCAN":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    else:
        raise ValueError("Algoritmo de clustering no soportado.")
    
    labels = clusterer.fit_predict(reduced_embeddings)
    return reduced_embeddings, labels

def plot_clusters(reduced_embeddings, labels):
    """Grafica los clusters en 2D."""
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", alpha=0.5)
    plt.colorbar()
    plt.title("Clusters en el Espacio Reducido")
    plt.show()


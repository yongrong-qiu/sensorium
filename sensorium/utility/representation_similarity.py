import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as hierarchy


# Compute Pearson Correlation Distance for Artificial and Natural Responses
def compute_correlation_matrix(data):
    """
    Compute pairwise Pearson correlation matrix and convert to distance.
    For 2D numpy array features with a shape of (n,m), the shape of the output distance
    array is (n,n).
    """
    corr = np.corrcoef(data)  # Correlation matrix
    distance = 1 - corr       # Convert correlation to distance
    distance = (distance + distance.T) / 2  # Ensure symmetry
    np.fill_diagonal(distance, 0)  # Set diagonal to zero
    return distance

# Compute Cosine Distance for Feature Weights
def compute_cosine_distance(features):
    """
    Compute pairwise cosine distance matrix.
    For 2D numpy array features with a shape of (n,m), the shape of the output distance
    array is (n,n).
    """
    cosine_sim = cosine_similarity(features)
    distance = 1 - cosine_sim  # Convert similarity to distance
    # Clip small negative values caused by numerical errors
    distance = np.clip(distance, a_min=0, a_max=None)
    distance = (distance + distance.T) / 2  # Ensure symmetry
    np.fill_diagonal(distance, 0)  # Set diagonal to zero
    return distance

def square2vec(matrix):
    """Convert a square-form distance matrix to a vector-form distance vector."""
    triu_idx = np.triu_indices_from(matrix, k=1)
    return matrix[triu_idx]

def hierarchical_clustering(rdm):
    """
    Perform hierarchical clustering on a distance matrix (rdm).
    Return the reordered matrix
    """
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(rdm, method='ward')
    # Get the optimal ordering of features
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
    optimal_order = dendro['leaves']
    # Reorder the RDM using the hierarchical clustering order
    rdm_reordered = rdm[np.ix_(optimal_order, optimal_order)]
    return rdm_reordered
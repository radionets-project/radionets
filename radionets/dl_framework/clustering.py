import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform


def bgmmClustering(data, n_components: int = 10, n_init: int = 1):
    """Use Bayesian Gaussian Mixture Model for clustering data. The BGMM can
    reduce the number of components.

    Parameters
    ----------
    data: 2d-array
        data to be clustered
    n_components: int
        maximal number of components for the model
    n_init: int
        number of clustered models, best is selected

    Returns
    -------
    bgmm:
        Bayesian Gaussian Mixture Model
    """
    bgmm = BayesianGaussianMixture(
        n_components=n_components, n_init=n_init, init_params="k-means++"
    ).fit(data)
    return bgmm


def gmmClustering(
    data, n_components: int = 10, n_init: int = 1, score_type: str = None
):
    """Use Gaussian Mixture Model for clustering data. The number of components
    is determined by the BIC or AIC score.

    Parameters
    ----------
    data: 2d-array
        data to be clustered
    n_components: int
        number of components for the model, selecting a score_type can reduce it
    n_init: int
        number of clustered models, best is selected
    score_type: str
        score to get best model with components < n_components. AIC or BIC

    Returns
    -------
    best_gmm:
        best fitted Gaussian Mixture Model
    """
    lowest_score = np.infty
    score = []
    if score_type:
        for i in range(n_components):
            gmm = GaussianMixture(
                n_components=i + 1, n_init=n_init, init_params="k-means++"
            ).fit(data)

            if score_type == "AIC":
                score.append(gmm.aic(data))
                if score[-1] < lowest_score and score[-1] > 0:
                    lowest_score = score[-1]
                    best_gmm = gmm
            elif score_type == "BIC":
                score.append(gmm.bic(data))
                if score[-1] < lowest_score:
                    lowest_score = score[-1]
                    best_gmm = gmm
    else:
        gmm = GaussianMixture(
            n_components=n_components, n_init=n_init, init_params="k-means++"
        ).fit(data)
    return best_gmm


def spectralClustering(data, n_components: int = None):
    """Use Spectral Clustering for clustering data.

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf

    Less complex decision for number of components is used.

    Parameters
    ----------
    data: 2d-array
        data points
    n_components: int
        number of components for the model

    Returns
    -------
    model:
        Spectral Clustering model
    """
    affinity_matrix = getAffinityMatrix(data)
    normalized_affinity_matrix = normalizeAffinityMatrix(affinity_matrix)
    w, _ = np.linalg.eigh(normalized_affinity_matrix)
    if n_components is None:
        # Important eigenvalues are equaling 1
        n_components = np.sum(w > 0.95)
        # print(f'Eigenvalues : {np.round(w, 3)}')

    model = SpectralClustering(n_clusters=n_components, affinity="precomputed").fit(
        affinity_matrix
    )
    return model


def getAffinityMatrix(coordinates, k: int = 7):
    """Calculate affinity matrix based on input coordinates matrix and the number
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour.

    Parameters
    ----------
    coordinates: 2d-array
        data points of shape (n, 2)
    k: int
        k nearest neighbour

    Returns
    -------
    affinity_matrix: 2d-array
        affinity matrix of shape (n, n)
    """
    dists = squareform(pdist(coordinates))

    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T

    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale

    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0

    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def normalizeAffinityMatrix(affinity_matrix):
    """Normalization of affinity matrix.

    Parameters
    ----------
    affinity_matrix: 2d-array
        affinity matrix to be normalized

    Returns
    -------
    L: 2d-array
        normalized affinity matrix
    """
    D = np.diag(np.sum(affinity_matrix, axis=1))
    D_inv = np.sqrt(np.linalg.inv(D))
    L = np.dot(D_inv, np.dot(affinity_matrix, D_inv))
    return L

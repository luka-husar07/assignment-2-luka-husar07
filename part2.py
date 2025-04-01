# import plotly.figure_factory as ff
# import plotly.figure_factory as ff
import math
import pickle
import time
import warnings
from itertools import cycle, islice
from pprint import pprint

import matplotlib.pyplot as plt
import myplots as myplt
import numpy as np
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn import cluster, datasets, mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from . import i_utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""


def fit_kmeans_sse(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    k: int,
    seed: int = 42,
) -> float:
    """Compute the SSE for a KMeans clustering of the dataset.

    Args:
        dataset: A tuple of three numpy arrays: X, y, centers.
        k: The number of clusters.
        seed: The random seed.

    Returns:
        The SSE for the KMeans clustering of the dataset.

    """
    # Is there a random_tate argument?
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    return 0.0  # replace by appropriate code


def fit_kmeans_inertia(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    k: int,
    seed: int = 42,
) -> float:
    """Compute the inertia for a KMeans clustering of the dataset.

    Args:
        dataset: A tuple of three numpy arrays: X, y, centers.
        k: The number of clusters.
        seed: The random seed.

    Returns:
        The inertia for the KMeans clustering of the dataset.

    """
    return 0.0  # replace by appropriate code


def compute() -> dict[str, Any]:
    """Compute the answers for Part 2.

    Returns:
        A dictionary of answers for Part 2.

    """
    answers = {}

    # A.	Call the make_blobs function with following parameters:
    # (center_box=(-20,20), n_samples=20, centers=5, random_state=12).

    # X, y, centers
    # data is a list of tuple or list of list of lists

    # Answer: return value from the make_blobs function in sklearn,
    # expressed as a list of three numpy arrays
    # Note: `make_blobs` sometimes returns only two values. Choose arguments
    # such that three values to be returned.
    answers["2A: blob"] = None

    # B. Modify the fit_kmeans function to return the SSE
    #   (see Equations 8.1 and 8.2 in the book).

    # Answer type: the `fit_kmeans` function
    answers["2B: fit_kmeans"] = None

    # C. Plot the SSE as a function of k for k=1,2,….,8, and choose the
    #   optimal k based on the elbow method.

    # Answer type: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    answers["2C: SSE plot"] = None

    # D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans
    # estimator called _inertia). Do the optimal k’s agree?

    # dct value has the same structure as in 2C
    answers["2D: inertia plot"] = None

    # dct value should be a string, e.g., "yes" or "no"
    answers["2D: do ks agree?"] = None

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)

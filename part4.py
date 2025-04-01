import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import Any, NDArray

# to calculate the clusters given the threshold
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_fct

# to smooth the data
from scipy.signal import savgol_filter
from sklearn import cluster

# import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from . import i_utils as u

# Part 4.
# Evaluation of Hierarchical Clustering over Diverse Datasets:
# In this task, you will explore hierarchical clustering over different
# datasets. You will also evaluate different ways to merge clusters
# and good ways to find the cut-off point for breaking the dendrogram.

# Fill these two functions with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def fit_hierarchical_cluster(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
    k: int,
) -> NDArray:
    return np.zeros([1, 1])  # Replace by correct return value


def get_distance_threshold(Z) -> tuple[dict[str, Any], dict[str, Any]]:
    """Get the distance threshold for the slope and curvature of the dendrogram.

    Args:
        Z (NDArray): The linkage matrix.

    Returns:
        tuple[dict, dict]: The distance threshold for the slope and curvature of the dendrogram.

    """
    return {}, {}  # Replace by correct dictonaries


def fit_modified(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
) -> tuple[NDArray, NDArray, dict, dict]:
    """Fit the modified hierarchical clustering.

    Args:
        dataset (tuple[NDArray]): The dataset.
        linkage (Literal["ward", "average", "complete", "single"]): The linkage type.

    Returns:
        tuple[NDArray, NDArray, dict, dict]: The modified hierarchical clustering.

    """
    return np.zeros([1, 1]), np.zeros([1, 1]), {}, {}  # Replace by correct values


def compute():
    answers = {}

    # A. Repeat parts 1.A and 1.B with hierarchical clustering. That is,
    # write a function called fit_hierarchical_cluster (or something similar)
    # that takes the dataset, the linkage type and the number of clusters,
    # that trains an AgglomerativeClustering sklearn estimator and returns the
    # label predictions. Apply the same standardization as in part 1.B. Use
    # the default distance metric (euclidean) and the default linkage (ward).

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    data = u.load_datasets(n_samples=100)
    answers["4A: datasets"] = None

    # Answer type:  the `fit_hierarchical_cluster` function
    answers["4A: fit_hierarchical_cluster"] = None

    # B. Apply your function from 4.A and make a plot similar to 1.C with
    # the four linkage types (single, complete, ward, centroid: rows in the
    # figure), and use 2 clusters for all runs. Compare the results to problem 1,

    # Create a pdf of the plots and return in your report.

    # Answer type: list of dataset abbreviations (see 1.C)
    # List the datasets that are correctly clustered that k-means could not handle
    answers["4B: cluster successes"] = None

    # C. There are essentially two main ways to find the cut-off point for
    # breaking the diagram: specifying the number of clusters and specifying
    # a maximum distance. The latter is challenging to optimize for without
    # knowing and/or directly visualizing the dendrogram, however, sometimes
    # simple heuristics can work well. The main idea is that since the merging
    # of big clusters usually happens when distances increase, we can assume
    # that a large distance change between clusters means that they should
    # stay distinct. Modify the function from part 1.A to calculate a cut-off
    # distance before classification. Specifically, estimate the cut-off
    # distance as the maximum rate of change of the distance between
    # successive cluster merges (you can use the scipy.hierarchy.linkage
    # function to calculate the linkage matrix with distances). Apply this
    # technique to all the datasets and make a plot similar to part 4.B.

    # Create a pdf of the plots and return in your report.

    # Answer type: the function described above in 4.C
    answers["4C: modified function"] = None

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)

# import plotly.figure_factory as ff
# import plotly.figure_factory as ff
import math
import pickle
import time
import warnings
from itertools import cycle, islice
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn import cluster, datasets, mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


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
    X, _, _ = dataset
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(X)
    sse = kmeans.inertia_  # Inertia is the SSE
    return sse


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
    X, _, _ = dataset
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(X)
    return kmeans.inertia_  # Return inertia directly


def compute_sse_plot(dataset):
    sse_values = []
    for k in range(1, 9):  # k from 1 to 8
        sse = fit_kmeans_sse(dataset, k=k)
        sse_values.append((k, sse))
    return sse_values


def compute_inertia_plot(dataset):
    inertia_values = []
    for k in range(1, 9):  # k from 1 to 8
        inertia = fit_kmeans_inertia(dataset, k=k)
        inertia_values.append((k, inertia))
    return inertia_values


def find_elbow_point(values):
    k = np.arange(1, len(values) + 1)
    values = np.array(values)

    # Create line between first and last point
    p1 = np.array([k[0], values[0]])
    p2 = np.array([k[-1], values[-1]])

    # Compute distances
    def distance(point, line_start, line_end):
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

    distances = [distance(np.array([k[i], values[i]]), p1, p2) for i in range(len(k))]
    return int(k[np.argmax(distances)])



def find_optimal_k(sse_values, inertia_values):
    k_values, sse = zip(*sse_values)
    _, inertia = zip(*inertia_values)

    # Find the elbow point for SSE
    elbow_k_sse = find_elbow_point(sse)

    # Find the elbow point for Inertia
    elbow_k_inertia = find_elbow_point(inertia)

    return elbow_k_sse, elbow_k_inertia


def plot_sse(sse_values, optimal_k):
    """Plot SSE as a function of k."""
    k_values, sse = zip(*sse_values)  # Unzip the k and SSE values
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sse, marker='o')
    plt.title('SSE vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(k_values)  # Set x-ticks to be the k values
    plt.grid()
    plt.axvline(x=optimal_k, color='r', linestyle='--', label='Optimal k')
    plt.legend()
    plt.savefig("sse_plot.pdf")  # Save the plot as a PDF
    plt.show()  # Show the plot


def plot_inertia(inertia_values, optimal_k):
    """Plot Inertia as a function of k."""
    k_values, inertia = zip(*inertia_values)  # Unzip the k and inertia values
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, marker='o')
    plt.title('Inertia vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)  # Set x-ticks to be the k values
    plt.grid()
    plt.axvline(x=optimal_k, color='r', linestyle='--', label='Optimal k')
    plt.legend()
    plt.savefig("inertia_plot.pdf")  # Save the plot as a PDF
    plt.show()  # Show the plot



def compute() -> dict[str, any]:
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
    X, y, centers = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12, return_centers=True)
    answers["2A: blob"] = [X, y, centers]

    # B. Modify the fit_kmeans function to return the SSE
    #   (see Equations 8.1 and 8.2 in the book).

    # Answer type: the `fit_kmeans` function
    answers["2B: fit_kmeans"] = fit_kmeans_sse

    # C. Plot the SSE as a function of k for k=1,2,….,8, and choose the
    #   optimal k based on the elbow method.

    # Answer type: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    sse_values = compute_sse_plot(answers["2A: blob"])
    answers["2C: SSE plot"] = sse_values
  # Call the plotting function

    # D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans
    # estimator called _inertia). Do the optimal k's agree?

    # dct value has the same structure as in 2C
    inertia_values = compute_inertia_plot(answers["2A: blob"])
    answers["2D: inertia plot"] = inertia_values

    optimal_k_sse, optimal_k_inertia = find_optimal_k(sse_values, inertia_values)
    plot_sse(sse_values, optimal_k_sse)
    plot_inertia(inertia_values, optimal_k_inertia)# Call the plotting function

    # dct value should be a string, e.g., "yes" or "no"
    optimal_k_sse, optimal_k_inertia = find_optimal_k(answers["2C: SSE plot"], answers["2D: inertia plot"])
    answers["2D: do ks agree?"] = "yes" if optimal_k_sse == optimal_k_inertia else "no"

    return answers



# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)

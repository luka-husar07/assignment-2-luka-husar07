import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from numpy.typing import NDArray

# to calculate the clusters given the threshold
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_fct

# to smooth the data
from scipy.signal import savgol_filter
from sklearn import cluster

# import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from part1 import fit_kmeans, compute as compute_part1

import utils as u

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
    X, _ = dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    return labels


def get_distance_threshold(Z) -> tuple[dict[str, Any], dict[str, Any]]:
    """Get the distance threshold for the slope and curvature of the dendrogram.

    Args:
        Z (NDArray): The linkage matrix.

    Returns:
        tuple[dict, dict]: The distance threshold for the slope and curvature of the dendrogram.

    """
    # Get the distances from Z
    distances = Z[:, 2]

    # Smooth the distances using Savitzky-Golay filter
    smoothed = savgol_filter(distances, window_length=5, polyorder=2)

    # Compute first and second derivatives
    slope = np.gradient(smoothed)
    curvature = np.gradient(slope)

    # Find the index of maximum slope and curvature
    max_slope_idx = np.argmax(slope)
    max_curvature_idx = np.argmax(np.abs(curvature))

    slope_thresh = distances[max_slope_idx]
    curvature_thresh = distances[max_curvature_idx]

    slope_info = {
        "index": int(max_slope_idx),
        "threshold": float(slope_thresh),
    }

    curvature_info = {
        "index": int(max_curvature_idx),
        "threshold": float(curvature_thresh),
    }

    return slope_info, curvature_info


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
    X, _ = dataset

    # Step 1: Compute linkage matrix
    Z = linkage_fct(X, method=linkage)

    # Step 2: Get distance thresholds
    slope_info, curvature_info = get_distance_threshold(Z)

    # Step 3: Apply fcluster using those thresholds
    k_slope = fcluster(Z, t=slope_info["threshold"], criterion="distance")
    k_curvature = fcluster(Z, t=curvature_info["threshold"], criterion="distance")

    return k_slope, k_curvature, slope_info, curvature_info 


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
    data = compute_part1()["1A: datasets"]
    results = {}

    for key in ["nc", "nm", "bvv", "add", "b"]:
        dataset = data[key]
        # You can choose a sensible value of k (e.g., from dataset[1] which holds ground-truth labels)
        true_labels = dataset[1]
        k = len(np.unique(true_labels))

        pred_labels = fit_hierarchical_cluster(dataset, linkage="ward", k=k)
        results[key] = pred_labels

    answers["4A: datasets"] = results

    # Answer type:  the `fit_hierarchical_cluster` function
    answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    # B. Apply your function from 4.A and make a plot similar to 1.C with
    # the four linkage types (single, complete, ward, centroid: rows in the
    # figure), and use 2 clusters for all runs. Compare the results to problem 1,

    # Create a pdf of the plots and return in your report.
    def plot_hierarchical_clusterings(data):
        linkage_types = ["single", "complete", "ward", "centroid"]
        k = 2
        n_rows = len(linkage_types)
        n_cols = len(data)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))

        for row_idx, linkage in enumerate(linkage_types):
            for col_idx, (name, (X, _)) in enumerate(data.items()):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if linkage == "centroid":
                    Z = linkage_fct(X_scaled, method="centroid")
                    labels = fcluster(Z, t=k, criterion="maxclust")
                else:
                    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = model.fit_predict(X_scaled)

                ax = axes[row_idx, col_idx]
                ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
                if row_idx == 0:
                    ax.set_title(name)
                if col_idx == 0:
                    ax.set_ylabel(f"{linkage} linkage")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig("hierarchical_vs_kmeans.pdf")
    plot_hierarchical_clusterings(data)

    # Answer type: list of dataset abbreviations (see 1.C)
    # List the datasets that are correctly clustered that k-means could not handle
    answers["4B: cluster successes"] = ['add']

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
    answers["4C: modified function"] = fit_modified

    def plot_modified_threshold_clusters(data):
        fig, axes = plt.subplots(2, len(data), figsize=(20, 8))
        for col_idx, (name, (X, _)) in enumerate(data.items()):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            dataset = (X_scaled, _)

            k_slope, k_curv, _, _ = fit_modified(dataset, linkage="ward")

            axes[0, col_idx].scatter(X[:, 0], X[:, 1], c=k_slope, cmap="coolwarm", s=25)
            axes[0, col_idx].set_title(f"{name} - slope")

            axes[1, col_idx].scatter(X[:, 0], X[:, 1], c=k_curv, cmap="coolwarm", s=25)
            axes[1, col_idx].set_title(f"{name} - curvature")

            axes[0, col_idx].set_xticks([]), axes[1, col_idx].set_xticks([])
            axes[0, col_idx].set_yticks([]), axes[1, col_idx].set_yticks([])

        plt.tight_layout()
        plt.savefig("modified_cutoff_clustering.pdf")

    plot_modified_threshold_clusters(data)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)

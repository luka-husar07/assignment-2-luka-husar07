import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def fit_kmeans(
    dataset: tuple[np.ndarray, np.ndarray],
    *,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    # Unpack the dataset
    data, _ = dataset
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, init='random', random_state=seed)
    kmeans.fit(data_scaled)
    
    # Return the predicted labels
    return kmeans.labels_


def compute() -> dict:
    """Compute answers for Part 1 questions.

    This function addresses the questions in Part 1 of the assignment,
    which involves loading datasets, implementing k-means clustering,
    and analyzing the results.

    Returns:
        A dictionary containing the answers for each question in Part 1.
        The keys of the dictionary are strings identifying the question,
        and the values are the corresponding answers.

    """
    answers = {}

    # A.	Load the following 5 datasets with 100 samples each: noisy_circles
    # (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly
    # distributed data (add), blobs (b). Use the parameters from
    # (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html),
    # with any random state. (with random_state = 42). Not setting the correct
    # random_state # will prevent me from checking your results.

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # Answer keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated names for the datasets)
    data = {
        "nc": make_circles(n_samples=100, noise=0.1, random_state=42),
        "nm": make_moons(n_samples=100, noise=0.1, random_state=42),
        "bvv": make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42),
        "add": make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42),
        "b": make_blobs(n_samples=100, centers=3, random_state=42)
    }
    answers["1A: datasets"] = data

    # B. Write a function called fit_kmeans that takes dataset (before any
    # processing on it), i.e., pair of (data, label) Numpy arrays, and
    # the number of clusters as arguments, and returns the predicted labels
    # from k-means clustering. Use the init='random' argument and make sure
    # to standardize the data (see StandardScaler transform), prior to fitting
    # the KMeans estimator. This is the function you will use in the following
    # questions.

    # dct value:  the `fit_kmeans` function
    # Example: Given a function:   `def fit_kmeans(...)`, return
    # .   answers["1B: fit_kmeans"] = fit_kmeans
    answers["1B: fit_kmeans"] = fit_kmeans

    # C. Make a big figure (4 rows x 5 columns) of scatter plots (where points
    # are colored by predicted label) with each column corresponding to the
    # datasets generated in part 1.A, and each row being k=[2,3,5,10] different
    # number of clusters. For which datasets does k-means seem to produce correct
    # clusters for (assuming the right number of k is specified) and for which
    # datasets # does k-means fail for all values of k?

    # Create a pdf of the plots and return in your report.

    # Answer: return a dictionary of the abbreviated dataset names (zero
    # or more elements) and associated k-values with correct clusters.
    # key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. The values are the
    # set of k for which there is success. Only return datasets where
    # the set of cluster sizes k is non-empty.
    # Provide any code and answer

    # 'add' almost works with k=3
    answers["1C: cluster successes"] = {
        'nc': {2},  # Noisy circles work well with k=2
        'nm': {2, 3},  # Noisy moons work well with k=2 and k=3
        'bvv': {3},  # Blobs with varied variances work well with k=3
        'b': {3},  # Blobs work well with k=3
    }

    # Answer: a list of dataset abbreviations (list has zero or more
    # elements, which are abbreviated dataset names as strings)
    answers["1C: cluster failures"] = ['add']  # Anisotropicly distributed data did not cluster well

    # D. Repeat 1.C a few times and comment on which (if any) datasets seem to
    # be sensitive to the choice of initialization for the k=2,3 cases.
    # You do not need to add the additional plots to your report.

    # Create a pdf of the plots and return in your report.
    #  STUDENT CODE

    def test_initialization_sensitivity(data, k_values=[2, 3], runs=10):
        sensitive_datasets = set()

        for name, (X, y_true) in data.items():
            for k in k_values:
                all_labels = []

                # Run k-means with different seeds
                for seed in range(runs):
                    labels = fit_kmeans((X, y_true), k=k, seed=seed)
                    all_labels.append(labels)

                # Compute pairwise ARI between all runs
                ari_scores = []
                for i in range(len(all_labels)):
                    for j in range(i+1, len(all_labels)):
                        ari = adjusted_rand_score(all_labels[i], all_labels[j])
                        ari_scores.append(ari)

                avg_ari = np.mean(ari_scores)

                # If ARI is consistently low, the clustering is unstable
                if avg_ari < 0.9:  # You can adjust this threshold
                    sensitive_datasets.add(name)
                    break  # No need to test k=3 if k=2 is already unstable

        return sorted(list(sensitive_datasets))


    # Answer: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    # All datasets appear insensitive to initialization when k=2, 3
    # When k=5, other datasets are sensitive to the initialization.
    sensitive = test_initialization_sensitivity(data)
    answers["1D: datasets sensitive to initialization"] = sensitive
    print(type(sensitive))


    # Call this function in compute() after fitting k-means
    plot_clusters(data, [2, 3, 5, 10])

    return answers


def plot_clusters(data, k_values):
    fig, axes = plt.subplots(len(k_values), len(data), figsize=(20, 16))
    
    for i, (key, (X, _)) in enumerate(data.items()):
        for j, k in enumerate(k_values):
            labels = fit_kmeans((X, _), k=k)
            axes[j, i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            axes[j, i].set_title(f"{key} - k={k}")
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig("report.pdf")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)

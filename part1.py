import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from . import i_utils as u

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
    return np.zeros(1)  # Replace by correct variable


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
    data = u.load_datasets(n_samples=100)
    answers["1A: datasets"] = None

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
    answers["1B: fit_kmeans"] = None

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
    answers["1C: cluster successes"] = None

    # Answer: a list of dataset abbreviations (list has zero or more
    # elements, which are abbreviated dataset names as strings)
    answers["1C: cluster failures"] = None

    # D. Repeat 1.C a few times and comment on which (if any) datasets seem to
    # be sensitive to the choice of initialization for the k=2,3 cases.
    # You do not need to add the additional plots to your report.

    # Create a pdf of the plots and return in your report.
    #  STUDENT CODE

    # Answer: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    # All datasets appear insensitive to initialization when k=2, 3
    # When k=5, other datasets are sensitive to the initialization.
    answers["1D: datasets sensitive to initialization"] = None

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)

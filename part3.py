
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean

# Part 3.
# Hierarchical Clustering:
# Recall from lecture that agglomerative hierarchical clustering is a greedy
# iterative scheme that creates clusters, i.e., distinct sets of indices of points,
# by gradually merging the sets based on some cluster dissimilarity (distance)
# measure. Since each iteration merges a set of indices there are at most n-1
# mergers until the all the data points are merged into a single cluster
# (assuming n is the total points). This merging process of the sets of indices
# can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative
# hierarchal clustering can be simply defined as a function that takes in a set of
# points and outputs the dendrogram.

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def compute():
    answers = {}

    # A. Load the provided dataset “hierarchical_toy_data.mat” using the scipy.io.loadmat function.

    # return value of scipy.io.loadmat()
    def load_data():
        data = io.loadmat('hierarchical_toy_data.mat')
        return data['X']  

    X = load_data()
    answers["3A: toy data"] = X

    # B. Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and
    # scipy.hierachy.dendrogram functions, with “single” linkage.
    # Include the dendrogram plot in your report.
    def create_dendrogram(X):
        Z = linkage(X, method='single')  # Using single linkage
        plt.figure(figsize=(10, 7))
        dendogram_data = dendrogram(Z)
        plt.title('Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig("dendrogram.pdf")  # Save the dendrogram as a PDF
        plt.show()  # Show the plot
        return Z, dendogram_data
    
    Z, dendogram_data = create_dendrogram(X)

    # Answer type: NDArray
    answers["3B: linkage"] = Z

    # Answer type: the return value of the dendogram function, which is of type `dict
    answers["3B: dendogram"] = dendogram_data

    # C. Consider the merger of the cluster corresponding to points with
    # index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0)
    # were these clusters merged? That is, what row does the merger of A
    # correspond to in the linkage matrix Z? The rows count from zero.
    def find_merging_iteration(Z, cluster_I, cluster_J):
        from collections import defaultdict

        # Step 1: Initialize dictionary to track cluster contents
        n = Z.shape[0] + 1  # number of original points
        clusters = {i: [i] for i in range(n)}  # each point starts in its own cluster

        # Step 2: Iterate through Z and update clusters
        for i, (a, b, _, _) in enumerate(Z):
            a, b = int(a), int(b)
            
            # Form new cluster containing all points in a and b
            new_cluster = clusters[a] + clusters[b]
            clusters[n + i] = new_cluster  # new cluster index is n + i

            # Check if this new cluster is exactly cluster_I + cluster_J
            if set(new_cluster) == set(cluster_I + cluster_J):
                return i  # found the merge step

        return None  # not found


    cluster_I = [8, 2, 13]
    cluster_J = [1, 9]
    # Answer type: integer
    answers["3C: iteration"] = find_merging_iteration(Z, cluster_I, cluster_J)


    # D.	Write a function that takes the data and the two index sets {I,J} above,
    # and returns the dissimilarity given by single link clustering using the Euclidian
    # distance metric. The function should output the same value as the 3rd column of the
    # row found in problem 2.C.

    # Answer type: a function defined above
    answers["3D: function"] = single_link_dissimilarity
    answers["3D: min_dist"] = single_link_dissimilarity(X, cluster_I, cluster_J)

    # E.	In the actual algorithm, deciding which clusters to merge should consider all
    # of the available clusters at each iteration. List all the clusters as index sets,
    # using a list of lists, e.g., [[0,1,2],[3,4],[5],[6],…],  that were available when the
    # two clusters in part 2.D were merged.

    def get_clusters_before_iteration(Z, merge_iteration):
        n = Z.shape[0] + 1  # number of original data points
        clusters = {i: [i] for i in range(n)}  # initially, each point is its own cluster

        # This will store the current set of active clusters
        active_clusters = {i for i in range(n)}

        for i in range(merge_iteration):
            a, b = int(Z[i, 0]), int(Z[i, 1])
            new_cluster_id = n + i

            # Merge contents of cluster a and b
            merged = clusters[a] + clusters[b]
            clusters[new_cluster_id] = merged

            # Update active clusters
            active_clusters.discard(a)
            active_clusters.discard(b)
            active_clusters.add(new_cluster_id)

        # Return the actual point indices in each cluster
        return [clusters[cid] for cid in active_clusters]


    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    merge_iteration = find_merging_iteration(Z, cluster_I, cluster_J)
    answers["3E: clusters"] = get_clusters_before_iteration(Z, merge_iteration)

    # F.	Single linked clustering is often criticized as producing clusters where
    # “the rich get richer”, that is, where one cluster is continuously merging with
    # all available points. Does your dendrogram illustrate this phenomenon?

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = (
    "Yes, the dendrogram clearly illustrates the 'rich get richer' phenomenon. "
    "On the right side, one cluster starts small and keeps merging with nearby points or clusters at very small distances, "
    "forming a long chain. This is a typical chaining effect of single linkage clustering, where one cluster gradually "
    "absorbs others rather than forming balanced merges."
)


    return answers

# Function for 3D
def single_link_dissimilarity(data, cluster_I, cluster_J):
        min_dist = float('inf')
        
        for i in cluster_I:
            for j in cluster_J:
                dist = euclidean(data[i], data[j])
                if dist < min_dist:
                    min_dist = dist

        return min_dist


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()
    print(answers["3E: clusters"])

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)

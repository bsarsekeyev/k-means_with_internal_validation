"""
K-Means implementation with Maximin initialization and Davies-Boldin Internal validation
"""
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")


class K_means:
    def __init__(self, k, iterations, tolerance):
        self.k = k
        self.iterations = iterations
        self.tolerance = tolerance

    # Eucludean Distance Calculator for centroid initializatoin

    def compute_initial_distance(self, centroid, X):
        return ((X-centroid)**2).sum(axis=1)

    # Compute Maximin Initialization Method
    def initialize_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        centroids[0] = X[np.random.randint(len(X)), :]
        distances = self.compute_initial_distance(centroids[0], X)
        for i in range(1, self.k):
            centroids[i] = X[np.argmax(distances)]
            distances = np.minimum(
                distances, self.compute_initial_distance(centroids[i], X))
        return centroids

    # Compute new centroids
    def compute_new_centroids(self, X, cluster):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            # new centroids based on the mean of the points in a cluster
            centroids[i, :] = np.mean(X[cluster == i, :], axis=0)
        return centroids

    # Euclidean distance calculator
    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.k))
        # calculate distance between centroids and each point
        for i in range(self.k):
            each_point = ((X-centroids[i, :])*(X-centroids[i, :])).sum(1)
            distance[:, i] = each_point
        return distance

    # Assign to the closest cluster
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    # Sum of the Squared Error
    def compute_sse(self, X, cluster, centroids):
        distance = np.zeros(X.shape[0])
        for i in range(self.k):
            distance[cluster == i] = (
                (X[cluster == i] - centroids[i])*(X[cluster == i] - centroids[i])).sum(1)
        return np.sum(distance)

    # Calculate the number of points that belong to each cluster
    def belongs_to(self, X, cluster):
        clusters = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            for j in range(X.shape[0]):
                if i == cluster[j]:
                    clusters[j, i] = 1
        # return the number of points that belong to each cluster
        return np.sum(clusters, axis=0)

    # Compute Davies - Bouldin Internal Validation Index
    def Davies_Bouldin(self, X, cluster):
        cluster_k = [X[cluster == k] for k in range(self.k)]
        centroids = [np.mean(k, axis=0) for k in cluster_k]
        dispersion = [np.mean([np.sum(x-centroids[i])**2 for x in k])
                      for i, k in enumerate(cluster_k)]

        db = []
        for i in range(self.k):
            for j in range(self.k):
                if j != i:
                    db.append(
                        (dispersion[i] + dispersion[j] / np.sum(centroids[i] - centroids[j])**2))
        return np.max(db)/self.k

    # Convergance
    def converge(self, X):
        self.centroids = self.initialize_centroids(X)
        old_sse = 0.0
        print("- Clustering -")

        # SSE iterations
        for i in range(self.iterations):
            old_centroids = self.centroids
            distance = self. compute_distance(X, old_centroids)
            self.cluster = self.find_closest_cluster(distance)
            self.centroids = self.compute_new_centroids(X, self. cluster)
            self.error = self. compute_sse(X, self.cluster, self.centroids)

            if old_sse == 0:
                print("Iteration 1 - obj: " + str(self.error)+"; delta obj: 0")
            else:
                print("Iteration "+str(i+1)+" - obj: " + str(self.error) +
                      "; delta obj:" + str((old_sse-self.error)/old_sse))

            # Find Empty Clusters in the iteration
            self.empty_cluster = self.belongs_to(X, self.cluster)
            for j in range(self.k):
                if self.empty_cluster[j] == 0:
                    print("Empty cluster: "+str(j+1))
                    # find the point that contributes the most to SSE
                    biggest_SSE_contributor = X[np.argmax(
                        np.sum(distance, axis=1)), :]
                    self.centroids[j] = biggest_SSE_contributor
            if(old_sse != 0.0):
                # Check convergence by comparing to tolerance
                if (old_sse - self.error)/old_sse < self.tolerance:
                    break
            old_sse = self.error
        self.db_index = self.Davies_Bouldin(X, self.cluster)
        print("Davies-Bouldin Index ("+str(self.db_index)+")")
        print()


def main():
    # Command Line Arguments
    F = sys.argv[1]  # Name of the dataset file
    I = int(sys.argv[2])  # Number of iterations
    T = float(sys.argv[3])  # Tolerance
    R = int(sys.argv[4])  # Number of Runs
    N = int(sys.argv[5])  # Normalization method choice <between 0 and 2>

    # Reading data into dataframe and converting to numpy array
    # parameters for read_csv should be changed relative to the dataset
    data = pd.read_csv(F, sep=" ")
    data = data.values
    data = data[:, 0:4]  # discarding labels if present for iris dataset
    data_mm = (data - np.min(data))/(np.max(data) - np.min(data))
    data_z = (data - np.mean(data))/np.std(data)

    if N == 0:
        X = data
    elif N == 1:
        X = data_mm
    elif N == 2:
        X = data_z
    else:
        print("Enter 0 for no normalization; 1 for Min-Max normalization; 2 for Z-score normalization")
    print(X)

    # find the maximum number of clusters for iris dataset
    max_k = np.rint(np.sqrt(X.shape[0]/2))
    max_k = int(max_k)
    print("Maximum numbers of clusters: "+str(max_k))

    # Keeping track of the best run and validation index
    best_db = []

    for i in range(2, max_k+1):
        # Instantiate a k-means class
        km = K_means(i, I, T)
        # Keeping track of the best run and validation index
        run = []
        db = []
        for j in range(R):
            print("Run: " + str(j+1))
            km.converge(X)
            run.append(km.error)
            db.append(km.db_index)

        print("-------------------")
        print("Best of " + str(R) + " Runs with " + str(i)+" clusters: " +
              str(run[run.index(min(run))])+" Davies-Bouldin Index : " + str(db[run.index(min(run))]))
        print("-------------------")

        best_db.append(db[run.index(min(run))])

    np.savetxt("Davies-Bouldin_index_outputs.txt",
               best_db, delimiter="\t", fmt="%f")


if __name__ == "__main__":
    main()

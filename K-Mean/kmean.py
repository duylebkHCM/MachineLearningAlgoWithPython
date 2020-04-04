from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

def kmeans_init_centroids(X,k):
    return X[np.random.choice(X.shape[0],k,replace=False)]

def kmeans_assign_labels(X,centroids):
    D = cdist(X, centroids)

    return np.argmin(D, axis = 1)

def has_converged(centroids, new_centroids):
    return (set([tuple(a) for a in centroids])) == (set([tuple(a) for a in new_centroids]))

def kmean_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k ,:]
        centroids[k,:] = np.mean(Xk, axis = 0)
    return centroids

def Kmeans(X,K):
    centroids = [kmeans_init_centroids(X,K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmean_update_centroids(X, labels[-1], K)
        if has_converged(new_centroids, centroids[-1]):
            break
        centroids.append(new_centroids)
        it+=1
    return (centroids, labels, it)

def Kmean_sklearn(X,K):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = K, random_state = 0).fit(X)
    print('Centers found by scikitlearn:')
    print(model.cluster_centers_)

def main():
    np.random.seed(18)
    means = [[2,2],[8,3],[3,6]]
    cov = [[1,0],[0,1]]

    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    X = np.concatenate((X0,X1,X2), axis = 0)
    K = 3

    original_label = np.asarray([0]*N+[1]*N+[2]*N).T

    centroids, labels, it = Kmeans(X,K)
    print('Centers found by our algorithm: \n', centroids[-1])

    Kmean_sklearn(X,K)

if __name__ == '__main__':
    main()
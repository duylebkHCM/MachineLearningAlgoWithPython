from __future__ import print_function
import numpy as np
from sklearn import datasets

mnist = datasets.load_digits()
print("Shape of minst data:", mnist.data.shape)

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
K = 10 # number of clusters
N = 10000
X = mnist.data[np.random.choice(mnist.data.shape[0], N)]
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

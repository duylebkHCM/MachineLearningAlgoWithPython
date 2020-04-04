import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
img = mpimg.imread('girl.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

for K in [2,5,10,15,20]:
    kmeans = KMeans(n_clusters=K).fit(X)
    labels = kmeans.predict(X)

    img_1 = np.zeros_like(X)
    for k in range(K):
        img_1[labels == k] = kmeans.cluster_centers_[k]

    img_2 =  img_1.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img_2, interpolation = 'nearest')
    plt.axis('off')
    plt.show()
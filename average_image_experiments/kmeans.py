import numpy as np
import os, pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_mnist_data(data_dir="/home/piyush/documents/programming/mnist/data/"):
    with open(os.path.join(data_dir, "X.p"), "rb") as f, \
         open(os.path.join(data_dir, "y.p"), "rb") as g:
        X = pickle.load(f)
        y = pickle.load(g)

    return X, y

def display_mnist_image(images, resolution=(28, 28)):
    dim = np.sqrt(len(images))
    if dim == int(dim):
        rows = columns = int(dim)
    else:
        dim = int(dim)
        rows, columns = dim, dim + 1

    fig = plt.figure()
    for i, image in enumerate(images):
        if image.shape != resolution:
            image = image.reshape((28, 28))

        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(image, cmap="gray")

    plt.show()

def mnist_kmeans(X, k=10, cache=False):
    if cache and os.path.exists("./mnist_kmeans_clusters_k%i.p" % k):
        with open("./mnist_kmeans_clusters_k%i.p" % k, "rb") as f:
            clusters = pickle.load(f)
    else:
        kmeans = KMeans(n_clusters=k, n_jobs=-1)
        clusters = kmeans.fit(X).cluster_centers_

        if cache:
            with open("./mnist_kmeans_clusters_k%i.p" % k, "wb") as f:
                pickle.dump(clusters, f)

    return clusters

if __name__ == "__main__":
    X, _ = get_mnist_data()
    clusters = mnist_kmeans(X, cache=True)
    display_mnist_image(clusters)

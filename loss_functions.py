import numpy as np


def kmeans_loss(x, cluster_centers, labels):
    loss = 0.0
    for i in range(len(labels)):
        a = np.array(x[i])
        b = cluster_centers[labels[i]]
        loss += np.linalg.norm(a - b)
    return loss

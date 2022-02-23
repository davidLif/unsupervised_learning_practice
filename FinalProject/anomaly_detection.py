import time

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances

anomaly_detect_algs = ["DBSCAN_anomaly", "IsolationForest"]
algo_types_anomaly_params = {
    "DBSCAN_anomaly": {"eps": [7, 10, 13, 15], "minSamples": [2, 4, 5]},
    "IsolationForest": {"contamination": [0.001, 0.01, 0.1]},
    "DirectDistance": {"maxDistance": [0.2, 0.25, 0.3], "K_neighbor": [2, 5, 10]}
}
# 5 diff methods:
# 1- Statistical model (Example: MAD)  - assumes normal distribution. Fails
# 2 - Clustering  ("bad grade" == anomaly), (one class SVM), (Isolation forest)
# 3 - Distance (DirectDistance)
# 4 - Density  (Density calc for each point - some formula over neighbors)
# 5 - Encoders

# https://medium.com/learningdatascience/anomaly-detection-techniques-in-python-50f650c75aaf

# https://donernesto.github.io/blog/outlier-detection-mad/
# https://donernesto.github.io/blog/outlier-detection-with-dbscan/
# https://donernesto.github.io/blog/outlier-detection-isolation-forest/

# Calculate using GMM visibility (must know num of clusters ahead of time)
# silhouette for point
# distance to center in Kmeans (must know num of clusters ahead of time)


def similarity(x, y):
    z = np.stack((x, y), axis=0)
    c = pairwise_distances(z)
    return c[0, 1]


def apply_anomaly_detection(data_matrix, alg, hyper_params_config):
    """

    :param data_matrix:
    :param alg:
    :param hyper_params_config:
    :return: new_labels - if label < , then the point is an outlier
    """
    if alg == "DBSCAN_anomaly":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["minSamples"]
                       , metric=similarity)

        s = time.time()
        new_labels = model.fit_predict(data_matrix)
        print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=time.time() - s))
        print("For hyper-params {0}".format(str(hyper_params_config)))
        new_labels[new_labels >= 0] = 0

        return new_labels, model
    elif alg == "IsolationForest":
        model = IsolationForest(contamination=hyper_params_config["contamination"])

        s = time.time()
        new_labels = model.fit_predict(data_matrix)
        print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=time.time() - s))
        new_labels[new_labels >= 0] = 0

        return new_labels, model
    elif alg == "DirectDistance":
        k = hyper_params_config["K_neighbor"] + 1  # + 1 helps us to ignore the (i,i) cell
        max_distance = hyper_params_config["maxDistance"]
        idx = np.argpartition(data_matrix, k)
        k_closest_index_to_row = idx.iloc[:, k].values
        new_labels = np.zeros((data_matrix.shape[0]))
        for i in range(data_matrix.shape[0]):
            k_neighbor_distance = data_matrix.iloc[i, k_closest_index_to_row[i]]
            if k_neighbor_distance <= max_distance:
                new_labels[i] = 0
            else:
                new_labels[i] = -1
        return new_labels, None
    else:
        raise Exception("no such anomaly detection algorithm")



import time
import numpy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

anomaly_detect_algs = ["DBSCAN_anomaly", "IsolationForest", "DirectDistance"]
algo_types_anomaly_params = {
    "DBSCAN_anomaly": {"eps": [1.5, 1.6, 1.7], "minSamples": [10, 20]},
    "IsolationForest": {"contamination": [0.01, 0.05, 0.1]},
    "DirectDistance": {"K_neighbor": [20, 30], "maxDistance": [1.5, 1.75, 2]}
}
# Optimal epsilon value for dbscan for clustering according to optimal_dbscan_e.py: 0.5
# Didn't work so well for anomaly detection :(

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


def apply_anomaly_detection(data_matrix, alg, hyper_params_config):
    """

    :param data_matrix:
    :param alg:
    :param hyper_params_config:
    :return: new_labels - if label < , then the point is an outlier
    """
    if alg == "DBSCAN_anomaly":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["minSamples"]
                       , metric="euclidean")

        s = time.time()
        new_labels = model.fit_predict(data_matrix)
        e = time.time()

        new_labels[new_labels >= 0] = 0
        anomaly_percentage = len(new_labels[new_labels < 0]) / len(new_labels)

        if anomaly_percentage <= 0.1:
            print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=e - s))
            print("For hyper-params {0}".format(str(hyper_params_config)))
            print("Anomaly percentage is {0}".format(anomaly_percentage))
        else:
            print("Non anomaly hyper-params {0}".format(str(hyper_params_config)))

        return new_labels, model
    elif alg == "IsolationForest":
        model = IsolationForest(contamination=hyper_params_config["contamination"])

        s = time.time()
        new_labels = model.fit_predict(data_matrix)
        e = time.time()

        new_labels[new_labels >= 0] = 0
        anomaly_percentage = len(new_labels[new_labels < 0]) / len(new_labels)

        if anomaly_percentage <= 0.1:
            print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=e - s))
            print("For hyper-params {0}".format(str(hyper_params_config)))
            print("Anomaly percentage is {0}".format(anomaly_percentage))
        else:
            print("Non anomaly hyper-params {0}".format(str(hyper_params_config)))

        return new_labels, model
    elif alg == "DirectDistance":
        k = hyper_params_config["K_neighbor"] + 1  # + 1 helps us to ignore the (i,i) cell
        max_distance = hyper_params_config["maxDistance"]

        new_labels = np.zeros((data_matrix.shape[0]))
        s = time.time()
        for i in range(data_matrix.shape[0]):
            distances_to_i = np.zeros((data_matrix.shape[0]))
            for j in range(data_matrix.shape[0]):
                distances_to_i[j] = numpy.linalg.norm(data_matrix[i]-data_matrix[j])
            sorted_distances = np.sort(distances_to_i)
            k_neighbor_distance = sorted_distances[k]
            if k_neighbor_distance <= max_distance:
                new_labels[i] = 0
            else:
                new_labels[i] = -1
        e = time.time()
        anomaly_percentage = len(new_labels[new_labels < 0]) / len(new_labels)

        if anomaly_percentage <= 0.1:
            print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=e - s))
            print("For hyper-params {0}".format(str(hyper_params_config)))
            print("Anomaly percentage is {0}".format(anomaly_percentage))
        else:
            print("Non anomaly hyper-params {0}".format(str(hyper_params_config)))

        return new_labels, None
    else:
        raise Exception("no such anomaly detection algorithm")



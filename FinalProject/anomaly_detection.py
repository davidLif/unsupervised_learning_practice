import time

from sklearn.cluster import DBSCAN

anomaly_detect_algs = ["DBSCAN"]
algo_types_anomaly_params = {
    "DBSCAN": {"eps": [0.1, 0.2, 0.3], "min_samples": [2, 4, 5]}
}
# 5 diff methods:
# 1- Statistical model (Example: MAD)
# 2 - Clustering  ("bad grade" == anomaly), (one class SVM), (Isolation forest)
# 3 - Distance (Example - kd-tree)
# 4 - Density  (Density calc for each point - some formula over neighbors)
# 5 - Encoders

# https://medium.com/learningdatascience/anomaly-detection-techniques-in-python-50f650c75aaf

# https://donernesto.github.io/blog/outlier-detection-mad/
# https://donernesto.github.io/blog/outlier-detection-with-dbscan/
# https://donernesto.github.io/blog/outlier-detection-isolation-forest/


def apply_anomaly_detection(distance_matrix, alg, hyper_params_config):
    """

    :param distance_matrix:
    :param alg:
    :param hyper_params_config:
    :return: new_labels - if label < , then the point is an outlier
    """
    if alg == "DBSCAN":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["min_samples"]
                       , metric='precomputed')
    else:
        raise Exception("no such anomaly detection algorithm")

    s = time.time()
    new_labels = model.fit_predict(distance_matrix)
    print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=time.time() - s))
    new_labels[new_labels >= 0] = 0

    return new_labels, model

import time

from sklearn.cluster import DBSCAN

anomaly_detect_algs = ["DBSCAN"]
algo_types_anomaly_params = {
    "DBSCAN": {"eps": [10, 0.1], "min_samples": [100, 10]}
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


def apply_anomaly_detection(x, alg, hyper_params_config):
    """

    :param x:
    :param alg:
    :param hyper_params_config:
    :return: new_labels - if label < , then the point is an outlier
    """
    if alg == "DBSCAN":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["min_samples"])
    else:
        raise Exception("no such anomaly detection algorithm")

    s = time.time()
    new_labels = model.fit_predict(x)
    print("Anomaly detection model {model} training seconds: {time}".format(model=alg, time=time.time() - s))

    return new_labels, model

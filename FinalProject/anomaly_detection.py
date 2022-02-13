anomaly_detect_algs = []

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
    if alg == "":
        pass
    return None

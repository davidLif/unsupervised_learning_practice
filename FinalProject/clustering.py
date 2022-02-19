import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from kmodes.kmodes import KModes
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import random_center_initializer, kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
# from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jaccard, euclidean, canberra

from pre_processing import compute_distances

n_clusters = [2, 5, 10]
distance_metrics = ["hamming", "jaccard"]
algo_types_clustering_params = {
    "DBSCAN": {"eps": [10, 0.1], "min_samples": [100, 10], "distance": distance_metrics},
    "Hierarchical": {"n_clusters": n_clusters, "linkage": ['average', 'single'],
                     "distance": distance_metrics},
    "KMeans": {"n_clusters": n_clusters},
    "KModes": {"n_clusters": n_clusters}
}
clustering_algs = list(algo_types_clustering_params.keys())


# currently returns only one or two clusters
# doesnt work well with other distance functions such as hamming/jaccard except for euclidean
class KMeans:
    """
    wrapper class for kmeans that support user defined distance metric
    (instead of sklearn kmeans that only support euclidean distance)
    """

    def __init__(self, data, n_clusters, distance_func=None):
        initial_centers = kmeans_plusplus_initializer(data, n_clusters, random_state=5).initialize()
        # instance created for respective distance metric
        self.instanceKm = kmeans(data, initial_centers=initial_centers,
                                 metric=distance_metric(type_metric.USER_DEFINED, func=distance_func))

    def fit_predict(self, data):
        # perform cluster analysis
        self.instanceKm.process()
        # cluster analysis results - clusters and centers
        pyClusters = self.instanceKm.get_clusters()
        # enumerate encoding type to index labeling to get labels
        pyEncoding = self.instanceKm.get_cluster_encoding()
        pyEncoder = cluster_encoder(pyEncoding, pyClusters, data)
        pyLabels = pyEncoder.set_encoding(0).get_clusters()
        return pyLabels


def apply_clustering(x, alg_type, hyper_params_config):
    if alg_type == "KMeans":
        model = KMeans(init="k-means++", n_clusters=hyper_params_config["n_clusters"])
    elif alg_type == "kmeans_adapted_distance":
        distance_func = jaccard
        model = KMeans(x, hyper_params_config["n_clusters"], distance_func)
    elif alg_type == "KModes":
        model = KModes(init="random", n_clusters=hyper_params_config["n_clusters"], n_jobs=-1)
    else:
        x = compute_distances(x, hyper_params_config["distance"])
        if alg_type == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=hyper_params_config["n_clusters"], affinity='precomputed',
                                            linkage=hyper_params_config["linkage"])
        elif alg_type == "DBSCAN":
            model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["min_samples"],
                           metric='precomputed')
        else:
            raise Exception("no such clustering algorithm")

    s = time.time()
    new_labels = model.fit_predict(x)
    print("Clustering model {model} training seconds: {time}".format(model=alg_type, time=time.time() - s))
    return new_labels, model


def train_eval_pipeline(data, model_name):
    pass  # TODO
    # algo_types_clustering_params[model_name]

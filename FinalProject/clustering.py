import time

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# !pip install kmodes
from kmodes.kmodes import KModes

n_clusters = [2, 3, 4, 5, 6, 7, 8]
# algo_types_clustering_params = {
#     "DBSCAN": {"eps": [0.1, 0.25, 0.5], "min_samples": [1, 5, 10]},
#     "Hierarchical": {"n_clusters": n_clusters},
#     "KMeans": {"n_clusters": n_clusters},
#     "KModes": {"n_clusters": n_clusters},
#     "GaussianMixture": {"n_components": n_clusters}
# }
n_clusters = [2, 5]
algo_types_clustering_params = {
    "DBSCAN": {"eps": [10, 0.1], "min_samples": [100, 10]},
    "Hierarchical": {"n_clusters": n_clusters},
    "KMeans": {"n_clusters": n_clusters},
    "KModes": {"n_clusters": n_clusters},
    "GaussianMixture": {"n_components": n_clusters}
}


def apply_clustering(x, alg_type, hyper_params_config):
    if alg_type == "KMeans":
        model = KMeans(init="k-means++", n_clusters=hyper_params_config["n_clusters"])
    elif alg_type == "KModes":
        model = KModes(init="random", n_clusters=hyper_params_config["n_clusters"], n_jobs=-1)
    elif alg_type == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=hyper_params_config["n_clusters"])
    elif alg_type == "DBSCAN":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["min_samples"])
    elif alg_type == "GaussianMixture":
        model = GaussianMixture(n_components=hyper_params_config["n_components"], random_state=0)
    else:
        raise Exception("no such clustering algorithm")

    s = time.time()
    new_labels = model.fit_predict(x)
    print(time.time()-s)
    return new_labels, model


def train_eval_pipeline(data, model_name):
    algo_types_clustering_params[model_name]

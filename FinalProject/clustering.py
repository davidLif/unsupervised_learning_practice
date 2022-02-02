from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


algo_types_clustering_params = {
    "DBSCAN": {"eps": [0.5, 1, 1.5], "min_samples": [15, 20, 25]},
    "Hierarchical": {"n_clusters": [3]},
    "KMeans": {"n_clusters": [3]}
    }


def apply_clustering(x, alg_type, hyper_params_config):
    if alg_type == "KMeans":
        model = KMeans(init="k-means++", n_clusters=hyper_params_config["n_clusters"])
    elif alg_type == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=hyper_params_config["n_clusters"])
    elif alg_type == "DBSCAN":
        model = DBSCAN(eps=hyper_params_config["eps"], min_samples=hyper_params_config["min_samples"])
    else:
        raise Exception("no such clustering algorithm")

    new_labels = model.fit_predict(x)

    return new_labels

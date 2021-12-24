import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split

from loss_functions import kmeans_loss

clustering_algo_types = ["KMeans", "Hierarchical", "DBSCAN"]
dimension_reduction_algo_types = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]
dimension_reduction_after_clustering = {
    "PCA": True,
    "CMDS": True,
    "ISO": False,
    "LLE": False,
    "EigenMaps": False
}
algo_types_clustering_params = {
    "DBSCAN": {
        "PCA": {"eps": 1, "min_samples": 19},
        "CMDS": {"eps": 4, "min_samples": 8},
        "ISO": {"eps": 10, "min_samples": 22},
        "LLE": {"eps": 0.01, "min_samples": 10},
        "EigenMaps": {"eps": 0.0007, "min_samples": 5}
    },
    "Hierarchical": {
        "PCA": {"n_clusters": 3},
        "CMDS": {"n_clusters": 3},
        "ISO": {"n_clusters": 3},
        "LLE": {"n_clusters": 3},
        "EigenMaps": {"n_clusters": 3}
    },
    "KMeans": {
        "PCA": {"dim_reduction_after": [True, False], "n_clusters": range(2, 5), "n_init": range(3, 6)},
        "CMDS": {"n_clusters": 3, "n_init": 4},
        "ISO": {"n_clusters": 3, "n_init": 4},
        "LLE": {"n_clusters": 3, "n_init": 4},
        "EigenMaps": {"n_clusters": 3, "n_init": 4}
    }
}


def load_data(num_records_per_class=200):
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    all_train_sets = []
    classes = [5.0, 6.0, 9.0]
    for cls in classes:
        idxs = labels == cls
        x_train = data[idxs][:num_records_per_class]
        y_train = labels[idxs][:num_records_per_class]
        all_train_sets.append(np.concatenate([x_train, np.expand_dims(y_train, 1)], axis=1))
    all_train_sets = np.concatenate(all_train_sets, axis=0)
    data_df = pd.DataFrame(all_train_sets)
    data_df.columns = np.append([f"feature{i}" for i in range(n_features)], 'label')

    print(f'num features: {len(data_df.columns) - 1}\nSize of the dataframe: {data_df.shape}')
    return data_df, classes


def dim_reduction(x, alg_type, n_components=2, k=5):
    if alg_type == "PCA":
        x = StandardScaler().fit_transform(x)  # normalizing the features
        model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "CMDS":
        model = MDS(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "ISO":
        model = Isomap(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "EigenMaps":
        model = SpectralEmbedding(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_df


def apply_clustering(x, alg_type, combo_config):
    model_metadata = {}

    if alg_type == "KMeans":
        model = KMeans(init="k-means++", n_clusters=combo_config["n_clusters"], n_init=combo_config["n_init"])
        new_labels = model.fit_predict(x)
        model_metadata["cluster_centers"] = model.cluster_centers_
    elif alg_type == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=combo_config["n_clusters"])
        new_labels = model.fit_predict(x)
    elif alg_type == "DBSCAN":
        model = DBSCAN(eps=combo_config["eps"], min_samples=combo_config["min_samples"])
        new_labels = model.fit_predict(x)
    else:
        raise Exception("no such clustering algorithm")

    return new_labels, model_metadata


def visualize(data_df, target_names, label_data, title=f"alg_type on dataset_name", out=""):
    plt.figure(figsize=(12, 12))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "grey", "aqua", "silver", "olive"]
    assert len(target_names) <= len(colors)\
        , f"not enough colors for num labels. num labels {len(target_names)} {title}"
    for i, (label_n, color) in enumerate(zip(target_names, colors)):
        i = target_names[i]
        indices_to_keep = label_data == i
        plt.scatter(data_df.loc[indices_to_keep, 'principle_cmp1']
                    , data_df.loc[indices_to_keep, 'principle_cmp2'], c=color)  # , s=50
    plt.legend(target_names, prop={'size': 15})

    plt.savefig(out)
    plt.close()


def run_single_algo(data, alg_type, clstr_type, combo_config, k=50):
    x = data.loc[:, data.columns[:- 1]].values

    if combo_config["dim_reduction_after"]:
        new_labels, model_metadata = apply_clustering(x, clstr_type, combo_config)
        reduced_data = dim_reduction(x, alg_type, k=k)
    else:
        reduced_data = dim_reduction(x, alg_type, k=k)
        new_labels, model_metadata = apply_clustering(reduced_data.values, clstr_type, combo_config)
    return x, reduced_data, new_labels, model_metadata


def PCA_KMeans_silhouette_score_results():
    data, _ = load_data(num_records_per_class=240)
    train, test = train_test_split(data, test_size=0.2)
    alg_type = "PCA"
    clstr_alg = "KMeans"
    combo_config_ranges = algo_types_clustering_params[clstr_alg][alg_type]

    all_hyper_params_results = {}

    for dim_red_after in combo_config_ranges["dim_reduction_after"]:
        for n_clusters in combo_config_ranges["n_clusters"]:
            for n_init in combo_config_ranges["n_init"]:
                combo_config = {
                    "dim_reduction_after": dim_red_after,
                    "n_clusters": n_clusters,
                    "n_init": n_init
                }

                x, reduced_data, n_labels, model_metadata = run_single_algo(train, alg_type, clstr_alg, combo_config, 50)

                s_score = silhouette_score(x, n_labels)
                if dim_red_after:
                    loss = kmeans_loss(x, model_metadata["cluster_centers"], n_labels)
                else:
                    loss = kmeans_loss(reduced_data.values, model_metadata["cluster_centers"], n_labels)

                name = f"{clstr_alg}_{alg_type}_{str(combo_config)}"
                target_names = np.unique(n_labels)
                #visualize(reduced_data, target_names, n_labels, title=f"{name} on 5-6-9", out=f"{name}.svg")

                all_hyper_params_results[(dim_red_after, n_clusters, n_init)] = (s_score, loss)
                print(f"{name} Statistics:")
                print(f"silhouette_score: {s_score}")
                print(f"loss function: {loss}")
                print()

    # 1: For K-means, it is easy to see the n_init parameter has little influence. We can ignore it.




def run_all_combinations():
    data, target_names = load_data(num_records_per_class=200)
    for clstr_alg in clustering_algo_types:
        for alg_type in dimension_reduction_algo_types:
            combo_config = algo_types_clustering_params[clstr_alg][alg_type]

            x, reduced_data, n_labels, model_metadata = run_single_algo(data, alg_type, clstr_alg, combo_config, 50)

            name = f"{clstr_alg}_{alg_type}"
            visualize(reduced_data, np.unique(n_labels), n_labels, title=f"{name} on 5-6-9", out=f"{name}.svg")


def main():
    PCA_KMeans_silhouette_score_results()
    #check_dbscan_runner()
    #run_all_combinations()


if __name__ == '__main__':
    main()

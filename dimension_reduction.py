import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, load_digits

clustering_algo_types = ["KMeans", "Hierarchical", "DBSCAN"]
dimension_reduction_algo_types = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]
dimension_reduction_algo_types_with_neighbors = ["ISO", "LLE", "EigenMaps"]


def load_data(num_records=200):
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    all_train_sets = []
    classes = [5.0, 6.0, 9.0]
    for cls in classes:
        idxs = labels == cls
        x_train = data[idxs][:num_records]
        y_train = labels[idxs][:num_records]
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


def apply_clustering(x, labels, alg_type):
    if alg_type == "KMeans":
        model = KMeans(init="k-means++", n_clusters=3, n_init=4)
    elif alg_type == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=3)
    elif alg_type == "DBSACN":
        model = DBSCAN(eps=0.15, min_samples=5)
    else:
        raise Exception("no such clustering algorithm")
    new_labels = model.fit_predict(x, labels)
    return new_labels


def visualize(data_df, target_names, label_data, title=f"alg_type on dataset_name", out=""):
    figure = plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black"]
    assert len(target_names) < len(colors), "not enough colors for num labels"
    for i, (label_n, color) in enumerate(zip(target_names, colors)):
        if label_n == -1:
            i = -1
        indices_to_keep = label_data == i
        plt.scatter(data_df.loc[indices_to_keep, 'principle_cmp1']
                    , data_df.loc[indices_to_keep, 'principle_cmp2'], c=color, s=50)
    plt.legend(target_names, prop={'size': 15})

    plt.savefig(out)
    plt.close()


def run_single_algo(data, alg_type="PCA", clstr_type="KMeans", k=50):
    x = data.loc[:, data.columns[:- 1]].values
    labels = data['label']
    reduced_data = dim_reduction(x, alg_type, k=k)
    new_labels = apply_clustering(reduced_data.values, labels, clstr_type)
    return reduced_data, new_labels


def main():
    data, target_names = load_data(num_records=200)
    for clstr_alg in clustering_algo_types:
        reduction_results = {}
        if clstr_alg == "DBSCAN":
            target_names.append(-1)
        for alg_type in dimension_reduction_algo_types:
            reduced_data, n_labels = run_single_algo(data, alg_type, clstr_alg, 50)
            reduction_results[alg_type] = {"reduced_data": reduced_data, "labels": n_labels}
            if alg_type not in dimension_reduction_algo_types_with_neighbors:
                k = ""
            name = f"{clstr_alg}_{alg_type}"
            visualize(reduced_data, target_names, n_labels, title=f"{name} on 5-6-9", out=f"{name}.png")


if __name__ == '__main__':
    main()

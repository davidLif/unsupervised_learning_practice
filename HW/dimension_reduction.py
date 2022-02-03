import itertools
import json
import os

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
from tqdm import tqdm

from HW.statistic_tests import find_best_config_based_on_statistic_test

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
        "PCA": {"dim_reduction_after": [True, False], "eps": [0.5, 1, 1.5], "min_samples": [15, 20, 25]},
        "CMDS": {"dim_reduction_after": [True, False], "eps": [3, 4, 5], "min_samples": [7, 8, 9]},
        "ISO": {"dim_reduction_after": [True, False], "eps": [8, 10, 12], "min_samples": [10, 20, 30],
                "n_neighbors": [50]},
        "LLE": {"dim_reduction_after": [True, False], "eps": [0.001, 0.01, 0.1], "min_samples": [5, 10, 15],
                "n_neighbors": [50]},
        "EigenMaps": {"dim_reduction_after": [True, False], "eps": [0.0001, 0.0007, 0.01], "min_samples": [2, 5, 8],
                      "n_neighbors": [50]}
    },
    "Hierarchical": {
        "PCA": {"dim_reduction_after": [True, False], "n_clusters": [3]},
        "CMDS": {"dim_reduction_after": [True, False], "n_clusters": [3]},
        "ISO": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]},
        "LLE": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]},
        "EigenMaps": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]}
    },
    "KMeans": {
        "PCA": {"dim_reduction_after": [True, False], "n_clusters": [3]},
        "CMDS": {"dim_reduction_after": [True, False], "n_clusters": [3]},
        "ISO": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]},
        "LLE": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]},
        "EigenMaps": {"dim_reduction_after": [True, False], "n_clusters": [3], "n_neighbors": [50]}
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


def dim_reduction(x, alg_type, hyper_params_config, n_components=2):
    if alg_type == "PCA":
        x = StandardScaler().fit_transform(x)  # normalizing the features
        model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "CMDS":
        model = MDS(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "ISO":
        model = Isomap(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "EigenMaps":
        model = SpectralEmbedding(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_df


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


def visualize(data_df, target_names, label_data, title=f"alg_type on dataset_name", out=""):
    plt.figure(figsize=(12, 12))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "grey", "aqua", "silver", "olive"]
    assert len(target_names) <= len(colors) \
        , f"not enough colors for num labels. num labels {len(target_names)} {title}"
    for i, (label_n, color) in enumerate(zip(target_names, colors)):
        i = target_names[i]
        indices_to_keep = label_data == i
        plt.scatter(data_df.loc[indices_to_keep, 'principle_cmp1']
                    , data_df.loc[indices_to_keep, 'principle_cmp2'], c=color)  # , s=50
    plt.legend(target_names, prop={'size': 15})

    plt.savefig(out)
    plt.close()


def run_single_algo(data, alg_type, clstr_type, hyper_params_config):
    if hyper_params_config["dim_reduction_after"]:
        new_labels = apply_clustering(data, clstr_type, hyper_params_config)
        reduced_data = dim_reduction(data, alg_type, hyper_params_config)
    else:
        reduced_data = dim_reduction(data, alg_type, hyper_params_config)
        new_labels = apply_clustering(reduced_data.values, clstr_type, hyper_params_config)
    return reduced_data, new_labels


def run_on_data_subsets(alg_type, clstr_alg, data_subsets, combo_config):
    s_scores, mi_scores = [], []
    for data in tqdm(data_subsets):
        x = data.loc[:, data.columns[:- 1]].values
        reduced_data, n_labels = run_single_algo(x, alg_type, clstr_alg, combo_config)
        if len(np.unique(n_labels)) > 1:
            s_score = silhouette_score(x, n_labels)
        else:
            s_score = None
        s_scores.append(s_score)
        mi_scores.append(mutual_info_score(data['label'], n_labels))
    return s_scores, mi_scores


def run_all_hyper_parameters_combos(alg_type, clstr_alg, combo_config_keys, data_subsets,
                                    hyper_parameters_config_options):
    current_algo_reduc_hyper_params_results = {}
    for hyper_parameters_config in hyper_parameters_config_options:
        combo_config = {}
        for config_key_index in range(len(combo_config_keys)):
            combo_config[combo_config_keys[config_key_index]] = hyper_parameters_config[config_key_index]
        s_scores, mi_scores = run_on_data_subsets(alg_type, clstr_alg, data_subsets, combo_config)
        current_algo_reduc_hyper_params_results[tuple(hyper_parameters_config)] = {"s_score": s_scores,
                                                                                   "mi_score": mi_scores}
    return current_algo_reduc_hyper_params_results


def get_subsets(full_data, num_subsets):
    subsets = []
    for test_iteration in range(num_subsets):
        data, _ = train_test_split(full_data, test_size=0.5,
                                   random_state=test_iteration)  # choose data for this iteration
        subsets.append(data)
    return subsets


def full_run(quantiative_data_p="quantiative_data.json", best_quantiative_data_p="quantiative_data.json"):
    if os.path.exists(quantiative_data_p):
        with open(quantiative_data_p, "r") as f:
            all_hyper_params_results = json.load(f)
    else:
        full_data, target_names = load_data(num_records_per_class=250)
        num_of_iterations_for_statistical_analysis = 10
        data_subsets = get_subsets(full_data, num_of_iterations_for_statistical_analysis)
        all_hyper_params_results = {}
        for clstr_alg in tqdm(clustering_algo_types):
            all_hyper_params_results[clstr_alg] = {}
            for alg_type in dimension_reduction_algo_types:
                combo_config_ranges = algo_types_clustering_params[clstr_alg][alg_type]
                combo_config_keys = list(combo_config_ranges.keys())
                hyper_parameters_config_options = list(itertools.product(*combo_config_ranges.values()))
                current_algo_reduc_hyper_params_results = \
                    run_all_hyper_parameters_combos(alg_type, clstr_alg, combo_config_keys
                                                    , data_subsets, hyper_parameters_config_options)

                all_hyper_params_results[clstr_alg][alg_type] = current_algo_reduc_hyper_params_results

            print(all_hyper_params_results)

    # check hyper params per clstr-dim redc configuration
    if not os.path.exists("stats"):
        os.mkdir("stats")

    for clstr, values in all_hyper_params_results.items():
        for dim_rdc, sub_values in values.items():
            quantiative_data = sub_values
            hyper_parameters_names = str(list(algo_types_clustering_params[clstr][dim_rdc]))
            save_path = f"stats/{clstr}_{dim_rdc}_hyperparam.csv"
            best_key = find_best_config_based_on_statistic_test(quantiative_data, hyper_parameters_names
                                                                , quantitative_score="s_score"
                                                                , save_path=save_path)
            all_hyper_params_results[clstr][dim_rdc] = sub_values[best_key]
            print(f"hyper param: {best_key}")

        # check dim reduction algorithm per clstr
        save_path = f"stats/{clstr}_dim_reduction_cmp_mi.csv"
        quantiative_data = values
        best_key = find_best_config_based_on_statistic_test(quantiative_data, None
                                                            , quantitative_score="mi_score"
                                                            , save_path=save_path)
        all_hyper_params_results[clstr] = values[best_key]
        print(f"dim reduction: {best_key}")

    #compare between clstr algorithms
    save_path = f"stats/clustering_cmp.csv"
    best_key = find_best_config_based_on_statistic_test(all_hyper_params_results, None
                                                        , quantitative_score="mi_score"
                                                        , save_path=save_path
                                                        , best_k_to_save=3)
    print(best_key)




    # With the all_iterations_results_storage, we can now run the paired and anova tests


def main():
    full_run()


if __name__ == '__main__':
    main()

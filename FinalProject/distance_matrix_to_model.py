import itertools
import os
import pandas as pd
from anomaly_detection import anomaly_detect_algs, apply_anomaly_detection, algo_types_anomaly_params
from dim_reduction import dim_reduction_algs, apply_dim_reduction
from clustering import algo_types_clustering_params, clustering_algs, apply_clustering
from tqdm import tqdm
import joblib


def load_data(distance_data_file):
    data_df = pd.read_csv(distance_data_file)
    return data_df.set_index("Unnamed: 0")


def run_all_hyper_parameters_combos(data, alg, algs_type, hyper_params_config_keys, hyper_parameters_config_options):
    for hyper_parameters_config in hyper_parameters_config_options:
        combo_config = {}
        for config_key_index in range(len(hyper_params_config_keys)):
            combo_config[hyper_params_config_keys[config_key_index]] = hyper_parameters_config[config_key_index]

        n_labels, model = run_single_algo(data, alg, algs_type, combo_config)

        param_name_part = str(list(combo_config.values())).replace(", ", "_").replace("[", "").replace("]", "")
        model_file_name = "./MODELS/{0}_{1}_{2}.model".format(algs_type, alg, param_name_part)
        joblib.dump(model, model_file_name)  # save model to file

        labeling_to_index = pd.DataFrame(n_labels, index=data.index, columns=["cluster-labeling"])
        labeling_to_index.to_csv("./LABELS/labels.{0}_{1}_{2}.csv".format(algs_type, alg, param_name_part))


def run_single_algo(data, alg, alg_type, hyper_params_config):
    if alg_type == "clustering":
        results, model = apply_clustering(data, alg, hyper_params_config)
    elif alg_type == "dim_reduction":
        results, model = apply_dim_reduction(data, alg, hyper_params_config)
    elif alg_type == "anomaly_detect":
        results, model = apply_anomaly_detection(data, alg, hyper_params_config)
    else:
        raise Exception("unKnown alg type")

    return results, model


def get_algos_and_params():
    if algs_type == "clustering":
        algs = clustering_algs
        params_config = algo_types_clustering_params
    elif algs_type == "dim_reduction":
        algs = dim_reduction_algs
        params_config = {}
    elif algs_type == "anomaly_detect":
        algs = anomaly_detect_algs
        params_config = algo_types_anomaly_params
    else:
        raise Exception("unknown algs type")
    return algs, params_config


if __name__ == "__main__":
    algs_type = "anomaly_detect"
    distance_data_file_path = "./DATA/USCensus1990.distance_data.csv"

    data = load_data(distance_data_file_path)
    algs, params_config = get_algos_and_params()

    # Prepare environment
    if not os.path.exists("./MODELS"):
        os.mkdir("./MODELS")
    if not os.path.exists("./LABELS"):
        os.mkdir("./LABELS")

    for alg_name in tqdm(algs):
        hyper_params_config = params_config[alg_name]
        hyper_params_config_keys = list(hyper_params_config.keys())
        hyper_parameters_config_options = list(itertools.product(*hyper_params_config.values()))
        run_all_hyper_parameters_combos(data, alg_name, algs_type, hyper_params_config_keys, hyper_parameters_config_options)

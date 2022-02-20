import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score


def load_labels_file(file_path):
    labels_df = pd.read_csv(file_path)
    return labels_df


def load_distance_data(distance_data_file):
    data_df = pd.read_csv(distance_data_file)
    return data_df.set_index("Unnamed: 0")


def calculate_score(data1, data2, score_method="silhouette", model=None):
    if score_method == "silhouette":
        if len(np.unique(data2)) > 1:
            return silhouette_score(data1, data2, metric="precomputed")
        return -100000
    elif score_method == "silhouette_per_sample":
        if len(np.unique(data2)) > 1:
            return silhouette_samples(data1, data2, metric="precomputed")
        return np.full(data2.shape, -100000)
    elif score_method == "mi":
        return mutual_info_score(data1, data2)
    elif "ic" in score_method and model is not None:
        try:
            if score_method == "aic":
                return model.aic(data1, metric="precomputed")
            else:
                return model.bic(data1, metric="precomputed")
        except Exception as e:
            print(e)
            return -100000
    else:
        raise Exception("unknown score method")


def find_nth(string, substring, n):
    """
    Find the n-th appearance of a substring and string.
    :param string:
    :param substring:
    :param n:
    :return: The index in string.
    """
    if n == 1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)


if __name__ == "__main__":
    distance_data_file_path = "./DATA/USCensus1990.distance_data.csv"
    score_method = "silhouette"

    distance_df = load_distance_data(distance_data_file_path)

    if not os.path.exists("./SCORES"):
        os.mkdir("./SCORES")

    scores = {}

    for labels_file_name in os.listdir("./LABELS"):
        file_path = os.path.join("./LABELS", labels_file_name)

        labels_df = load_labels_file(file_path)
        labels_array = labels_df["cluster-labeling"].to_numpy()

        score = calculate_score(distance_df, labels_array, score_method)

        full_type_str = labels_file_name.replace("labels.", "").replace(".csv", "")
        algo_type = full_type_str.split("_")[0]
        algo = full_type_str.split("_")[1]
        algo_params = full_type_str[find_nth(full_type_str, "_", 2) + 1:]

        if algo_type not in scores:
            scores[algo_type] = {}
        if algo not in scores[algo_type]:
            scores[algo_type][algo] = {}
        if algo_params not in scores[algo_type][algo]:
            scores[algo_type][algo][algo_params] = {}

        scores[algo_type][algo][algo_params] = score

    for algo_type in scores:
        for algo in scores[algo_type]:
            scores_row = scores[algo_type][algo]

            df = pd.DataFrame([list(scores_row.values())], columns=list(scores_row.keys()))
            df.to_csv("./SCORES/scores.{0}_{1}_{2}.csv".format(algo_type, score_method, algo))






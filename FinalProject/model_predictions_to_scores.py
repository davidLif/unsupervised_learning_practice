import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score


def load_labels_file(file_path):
    labels_df = pd.read_csv(file_path)
    return labels_df



def calculate_score(data1, data2, score_method="silhouette", model=None):
    if score_method == "silhouette":
        if len(np.unique(data2)) > 1:
            return silhouette_score(data1, data2)
        return -100000
    elif score_method == "silhouette_per_sample":
        if len(np.unique(data2)) > 1:
            return silhouette_samples(data1, data2)
        return np.full(data2.shape, -100000)
    elif score_method == "mi":
        return mutual_info_score(data1, data2)
    elif "ic" in score_method and model is not None:
        try:
            if score_method == "aic":
                return model.aic(data1)
            else:
                return model.bic(data1)
        except Exception as e:
            print(e)
            return -100000
    else:
        raise Exception("unknown score method")


if __name__ == "__main__":
    for labels_file_name in os.listdir("./LABELS"):
        file_path = os.path.join("./LABELS", labels_file_name)

        labels_df = load_labels_file(file_path)
        labels_array = labels_df["cluster-labeling"]


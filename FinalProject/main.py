import os.path
import numpy as np
import pandas as pd
from evaluator import Evaluator, stats_dir
from pre_processing import compute_distances


def load_data():
    data_df = pd.read_csv("./DATA/USCensus1990.clean_data.csv", dtype=np.int)
    # train_data, test_data = train_test_split(data_df, test_size=0.2,
    #                                          random_state=0)  # choose data for this iteration
    return data_df.drop(columns="Unnamed: 0").sample(100, random_state=0)  # train_data, test_data


if __name__ == "__main__":
    data = load_data()

    # Prepare environment
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)

    clustering_evaluator = Evaluator(data, algs_type="clustering")
    clustering_evaluator.algs_evaluation_pipeline(save_best=True)

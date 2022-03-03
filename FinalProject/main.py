import os.path
import numpy as np
import pandas as pd
from evaluator import Evaluator, stats_dir
from visualization import plot_all_boxplots


def load_data():
    original_data_df = pd.read_csv("./DATA/USCensus1990.clean_data.csv", dtype=np.int)
    data_df = pd.read_csv("./DATA/USCensus1990.MCA_20features.csv", dtype=np.float)
    original_data_df = original_data_df.loc[data_df["Unnamed: 0"]]
    original_data_df.index = data_df.index
    return data_df.drop(columns="Unnamed: 0"), original_data_df.drop(columns="Unnamed: 0")#.sample(100, random_state=0)  # train_data, test_data


if __name__ == "__main__":
    data, original_data = load_data()

    # Prepare environment
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)

    clustering_evaluator = Evaluator(data, algs_type="clustering")
    best_model_path = clustering_evaluator.algs_evaluation_pipeline(save_best=True)
    clustering_evaluator = Evaluator(data, algs_type="anomaly_detect")
    best_model_path = clustering_evaluator.algs_evaluation_pipeline(save_best=True)
    dim_reduction_evaluator = Evaluator(data, do_one_hot=False, algs_type="dim_reduction", model_path=best_model_path)
    dim_reduction_evaluator.algs_evaluation_pipeline(save_best=True)
    # from pathlib import Path
    # plot_all_boxplots(Path("results/anomaly_evaluation"))

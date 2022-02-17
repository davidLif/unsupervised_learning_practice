import os.path
import pandas as pd
from evaluator import Evaluator, stats_dir


def load_data():
    data_df = pd.read_csv("./DATA/USCensus1990.clean_data.csv")
    # train_data, test_data = train_test_split(data_df, test_size=0.2,
    #                                          random_state=0)  # choose data for this iteration
    return data_df.sample(100, random_state=0)  # train_data, test_data


if __name__ == "__main__":
    #TODO: What is the launch folder of the project? FinalProject?
    data = load_data()

    # Prepare environment
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)

    clustering_evaluator = Evaluator(data, algs_type="anomaly_detect")
    clustering_evaluator.algs_evaluation_pipeline()

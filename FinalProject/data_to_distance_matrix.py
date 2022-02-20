import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

external_variables = ["dAge", "dHispanic", "iYearwrk", "iSex"]


def load_data():
    data_df = pd.read_csv("./DATA/USCensus1990.clean_data.csv")
    # train_data, test_data = train_test_split(data_df, test_size=0.2,
    #                                          random_state=0)  # choose data for this iteration
    return data_df.sample(100, random_state=0)  # train_data, test_data


def convert_data_to_distances_matrix(data_df):
    data_df_no_unnamed = data_df.drop(columns="Unnamed: 0")
    data_df_no_unnamed = data_df_no_unnamed.drop(columns=external_variables)
    distances_df = pairwise_distances(data_df_no_unnamed, metric="hamming")
    distances_df = pd.DataFrame(distances_df, index=data_df["Unnamed: 0"], columns=data_df["Unnamed: 0"])
    distances_df.to_csv("./DATA/USCensus1990.distance_data.csv")


if __name__ == "__main__":
    data = load_data()
    convert_data_to_distances_matrix(data)


import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc


def load_data():
    data_df = pd.read_csv("./DATA/USCensus1990.MCA_20features.csv")
    return data_df.drop(columns="Unnamed: 0").sample(100000, random_state=0)  # train_data, test_data


if __name__ == "__main__":
    external_variables = ["dAge", "dHispanic", "iYearwrk", "iSex"]

    data = load_data()

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data.drop(columns=external_variables))
    distances, indices = nbrs.kneighbors(data.drop(columns=external_variables))
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(8, 12))
    plt.xlabel('Data point index - Indecies are sorted by Y value', fontsize=20)
    plt.ylabel('Points distance', fontsize=20)
    plt.plot(distances)
    plt.savefig("epsilon_elbow_graph.svg")

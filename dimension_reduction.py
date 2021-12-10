import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml

dimension_reduction_algo_types = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]


def load_data():
    mnist = fetch_openml('mnist_784')
    number_9_x_train = mnist.data[mnist.target == "9"][:100]
    number_9_y_train_tmp = mnist.target[mnist.target == "9"][:100]
    number_6_x_train = mnist.data[mnist.target == "6"][:100]
    number_6_y_train_tmp = mnist.target[mnist.target == "6"][:100]

    number_9_y_train = np.zeros((number_9_y_train_tmp.shape[0], 1))
    i = 0
    for train_tuple in number_9_y_train_tmp.iteritems():
        number_9_y_train[i] = train_tuple[1]
        i += 1
    number_9_train = np.concatenate([number_9_x_train, number_9_y_train], axis=1)
    number_6_y_train = np.zeros((number_6_y_train_tmp.shape[0], 1))
    i = 0
    for train_tuple in number_6_y_train_tmp.iteritems():
        number_6_y_train[i] = train_tuple[1]
        i += 1
    number_6_train = np.concatenate([number_6_x_train, number_6_y_train], axis=1)
    numbers_train = np.concatenate([number_9_train, number_6_train], axis=0)

    data_df = pd.DataFrame(numbers_train)
    data_df.columns = np.append(mnist.feature_names, 'label')

    print(f'num features: {len(data_df.columns)}\nSize of the dataframe: {data_df.shape}')
    return data_df, data_df.columns, [6.0, 9.0]


def dim_reduction(data, alg_type, n_components=2):
    if alg_type == "PCA":
        x = data.loc[:, data.columns[:data.shape[1] - 1]].values
        x = StandardScaler().fit_transform(x)  # normalizing the features
        model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "CMDS":
        model = MDS(n_components=n_components)
        transformed_data = model.fit_transform(data)
    elif alg_type == "ISO":
        model = Isomap(n_components=n_components)
        transformed_data = model.fit_transform(data)
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=5, n_components=n_components)
        transformed_data = model.fit_transform(data)
    elif alg_type == "EigenMaps":
        model = SpectralEmbedding(n_components=n_components)
        transformed_data = model.fit_transform(data)
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_df


def visualize(data_df, target_names, label_data, title=f"alg_type on dataset_name", out=""):
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black"]
    assert len(target_names) < len(colors), "not enough colors for num labels"
    for label_n, color in zip(target_names, colors):
        indices_to_keep = label_data == label_n
        plt.scatter(data_df.loc[indices_to_keep, 'principle_cmp1']
                    , data_df.loc[indices_to_keep, 'principle_cmp2'], c=color, s=50)
    plt.legend(target_names, prop={'size': 15})

    plt.savefig(out)


def run_single_algo(alg_type="PCA"):
    data, labels, target_names = load_data()
    reduced_data = dim_reduction(data, alg_type)
    visualize(reduced_data, target_names, data['label'], out=f"{alg_type}.png")


if __name__ == '__main__':
    run_single_algo(dimension_reduction_algo_types[0])
    run_single_algo(dimension_reduction_algo_types[1])
    run_single_algo(dimension_reduction_algo_types[2])
    run_single_algo(dimension_reduction_algo_types[3])
    run_single_algo(dimension_reduction_algo_types[4])

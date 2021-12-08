import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

dimension_reduction_algo_types = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]


def load_data():
    breast_cancer = load_breast_cancer()
    x_train = breast_cancer.data
    y_train = breast_cancer.target
    features_labels = np.append(breast_cancer.feature_names, 'label')

    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    final_breast_data = np.concatenate([x_train, y_train], axis=1)

    data_df = pd.DataFrame(final_breast_data)

    data_df.columns = features_labels

    print(f'num features: {len(features_labels)}\nSize of the dataframe: {data_df.shape}')
    return data_df, features_labels, breast_cancer.target_names


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

    transformed_data_Df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_Df


def visualize(data_df, target_names, label_data, title = f"alg_type on dataset_name",out=""):
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

    data['label'].replace(0, 'benign', inplace=True)
    data['label'].replace(1, 'malignant', inplace=True)
    visualize(reduced_data, target_names, data['label'], "", f"{alg_type}.png")


if __name__ == '__main__':
    run_single_algo(dimension_reduction_algo_types[0])
    run_single_algo(dimension_reduction_algo_types[1])
    run_single_algo(dimension_reduction_algo_types[2])
    run_single_algo(dimension_reduction_algo_types[3])
    run_single_algo(dimension_reduction_algo_types[4])

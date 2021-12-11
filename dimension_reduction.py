import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, load_digits

dimension_reduction_algo_types = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]
dimension_reduction_algo_types_with_neighbors = ["ISO", "LLE", "EigenMaps"]


def load_data(num_records=200):
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    all_train_sets = []
    classes = [5.0, 6.0, 9.0]
    for cls in classes:
        idxs = labels == cls
        x_train = data[idxs][:num_records]
        y_train = labels[idxs][:num_records]
        all_train_sets.append(np.concatenate([x_train, np.expand_dims(y_train,1)], axis=1))
    all_train_sets = np.concatenate(all_train_sets, axis=0)
    data_df = pd.DataFrame(all_train_sets)
    data_df.columns = np.append([f"feature{i}" for i in range(n_features)], 'label')

    print(f'num features: {len(data_df.columns)-1}\nSize of the dataframe: {data_df.shape}')
    return data_df, classes


def dim_reduction(x, alg_type, n_components=2, k=5, norm=True):
    if norm:
        x = StandardScaler().fit_transform(x)  # normalizing the features
    if alg_type == "PCA":
        model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "CMDS":
        model = MDS(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "ISO":
        model = Isomap(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "EigenMaps":
        model = SpectralEmbedding(n_neighbors=k, n_components=n_components)
        transformed_data = model.fit_transform(x)
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_df


def visualize(data_df, target_names, label_data, title=f"alg_type on dataset_name", out=""):
    figure = plt.figure(figsize=(10, 10))
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
    plt.close()


def run_single_algo(data, target_names, alg_type="PCA", k=5, norm=True):
    x = data.loc[:, data.columns[:- 1]].values
    reduced_data = dim_reduction(x, alg_type, k=k, norm=norm)
    if alg_type not in dimension_reduction_algo_types_with_neighbors:
        k = ""
    name = f"{alg_type}_{k}" if norm else f"{alg_type}_{k}_unnormed"
    visualize(reduced_data, target_names, data['label'], title=f"{name} for 5-6-9 images db"
              , out=f"{name}.png")


def main():
    data, target_names = load_data(num_records=200)
    ks = [5, 10, 20, 50, 100]
    for alg_type in dimension_reduction_algo_types:
        if alg_type in dimension_reduction_algo_types_with_neighbors:
            for k in ks:
                run_single_algo(data, target_names, alg_type, k)
        else:
            run_single_algo(data, target_names, alg_type)


if __name__ == '__main__':
    main()

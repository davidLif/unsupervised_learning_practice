import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import MDS, LocallyLinearEmbedding


def load_data():
    x_train = None
    y_train = 0
    x_train_flat = x_train.reshape(-1, 3072)
    feat_cols = [f'feat{i}' for i in range(x_train_flat.shape[1])]
    data_df = pd.DataFrame(x_train_flat,columns=feat_cols)
    data_df['label'] = y_train
    print(f'num features: {len(feat_cols)}\nSize of the dataframe: {data_df.shape}')
    return data_df


def dim_reduction(data, alg_type, n_components=2):
    model = None
    if alg_type == "PCA":
        pass
    elif alg_type == "CMDS":
        model=MDS(n_components=n_components)
    elif alg_type == "ISO":
        pass
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=5, n_components=n_components)
    elif alg_type == "EigenMaps":
        pass
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data = model.fit_transform(data)

    return transformed_data


def visualize(data_df, labels_names, title = f"alg_type on dataset_name",out=""):
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black" ]
    assert len(labels_names) < len(colors), "not enough colors for num labels"
    for label_n, color in zip(labels_names, colors):
        indicesToKeep = data_df['label'] == label_n
        plt.scatter(data_df.loc[indicesToKeep, 'principle_cmp1']
                    , data_df.loc[indicesToKeep, 'principle_cmp2'], c=color, s=50)
    plt.legend(labels_names, prop={'size': 15})

    plt.savefig(out)


def main(args):
    alg_type = "PCA"  # args[1]
    data, labels = load_data()
    reduced_data = dim_reduction(data, alg_type)
    visualize(reduced_data, labels, f"{alg_type}.png")


if __name__ == '__main__':
    main(sys.argv)

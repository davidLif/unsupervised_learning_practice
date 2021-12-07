import sys
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    # return x and labels?
    pass


def dim_reduction(data, alg_type):
    model = None
    if alg_type == "PCA":
        pass
    elif alg_type == "CMDS":
        pass
    elif alg_type == "ISO":
        pass
    elif alg_type == "LLE":
        pass
    elif alg_type == "EigenMaps":
        pass
    else:
        raise Exception("unrecognized algorithm type")

    new_data = model.fit_transform(data)
    return new_data


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
    for target, color in zip(labels_names, colors):
        indicesToKeep = data_df['label'] == target
        plt.scatter(data_df.loc[indicesToKeep, 'dimension 1']
                    , data_df.loc[indicesToKeep, 'dimension 2'], c=color, s=50)

    plt.savefig(out)


def main(args):
    alg_type = "PCA"  # args[1]
    data, labels = load_data()
    reduced_data = dim_reduction(data, alg_type)
    visualize(reduced_data, labels, f"{alg_type}.png")


if __name__ == '__main__':
    main(sys.argv)

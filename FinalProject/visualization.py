import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_clusters(data, targets, save_path):
    df = pd.DataFrame()
    df['x'] = data.values[:, 0]
    df['y'] = data.values[:, 1]
    df["clustering_results"] = clustering_results
    num_targets = targets.shape[1]
    targets.index = df.index
    fig, subs = plt.subplots(nrows=1, ncols=num_targets, figsize=(16, 7))
    for i in range(num_targets):
        df['target'] = targets[i]
        sns.scatterplot(x='x', y='y', hue='target', palette=sns.color_palette("hls", 10), data=df,
                        legend="full", ax=subs[i])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_clusters(data, targets, clustering_results, save_path):
    df = pd.DataFrame()
    df['cmp 1'] = data.values[:, 0]
    df['cmp 2'] = data.values[:, 1]
    df["clustering_results"] = clustering_results
    num_targets = targets.shape[1]
    targets.index = df.index
    fig, subs = plt.subplots(nrows=2, ncols=num_targets, figsize=(16, 7))
    targets["clustering_results"] = clustering_results
    first_row = True
    for i in range(2):
        for j in range(num_targets):
            if first_row:
                j = int(num_targets / 2)
                ncol = 3
                bbox_to_anchor = (-0.3, 1)
            col_name = targets.columns[j] if not first_row else "clustering_results"
            df['target'] = targets[col_name]
            sub_df = df.sample(1000, random_state=0)
            sns.scatterplot(x='cmp 1', y='cmp 2', hue='target',
                            palette=sns.color_palette("hls", len(np.unique(sub_df['target']))),
                            data=sub_df, legend="full", ax=subs[i, j])
            subs[i, j].set_yticklabels([])
            subs[i, j].set_xticklabels([])
            subs[i, j].title.set_text(col_name)
            subs[i, j].legend(bbox_to_anchor=bbox_to_anchor,  # , loc="center left",
                              ncol=ncol)
            # subs[i, j].legend(ncol=ncol, loc="best")
            if first_row:
                for x in [0, 1, 3]:
                    fig.delaxes(subs[0, x])
                first_row = False
                ncol = 1
                bbox_to_anchor = (0.06, 0.9)
                break
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_all_dim_reduc(data, save_path):
    algs = set([x.split("_")[0] for x in data.columns if "cmp" in x])
    df = pd.DataFrame()
    df["clustering_results"] = data["clustering_results"]
    df["iYearwrk"] = data["iYearwrk"]
    fig, subs = plt.subplots(nrows=len(algs), ncols=2, figsize=(5, 2.5 * len(algs)), sharey="row")
    for i, dim_reduc_alg in enumerate(algs):
        df['cmp 1'] = data[f"{dim_reduc_alg}_cmp1"]
        df['cmp 2'] = data[f"{dim_reduc_alg}_cmp2"]
        sub_df = df.sample(1000, random_state=0)
        for j, hue in enumerate(["iYearwrk", "clustering_results"]):
            colors = sns.color_palette("hls", len(np.unique(sub_df[hue])))
            if hue == "clustering_results" and -1 in sub_df[hue].values:
                colors.insert(0, (0, 0, 0))
                colors = colors[:-1]
            sns.scatterplot(x='cmp 1', y='cmp 2', hue=hue,
                            palette=colors, data=sub_df,  ax=subs[i, j]) #legend="full",
            subs[i, j].tick_params(axis='both', which='both', labelsize=6)
            subs[i, j].set_title(f"{dim_reduc_alg}, {hue}", size=8)
            subs[i, j].set_xlabel("")
            subs[i, j].set_ylabel("")
            subs[i, j].legend([])#box_to_anchor=bbox_to_anchor)  # , loc="center left",)
            # subs[i, j].set_aspect('equal')
    fig.text(0.5, 0.04, "principle component 1", ha="center", va="center")
    fig.text(0.05, 0.5, "principle component 2", ha="center", va="center", rotation=90)
    handles, labels = subs[0,0].get_legend_handles_labels()
    legend1 = plt.legend(handles, labels,  bbox_to_anchor=(-1.5,5))#, loc='center')
    handles, labels = subs[0,1].get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.1,5))#, loc='center')
    plt.gca().add_artist(legend1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_mi_boxplots(csv_fs, save_path):
    """
    plot the silhouette distribution using boxplot
    :param csv_fs: list of all the csv files we want to plot theirs boxplot
    :param save_path: where to save the plot
    :return: None
    """
    fig, subs = plt.subplots(1, len(csv_fs), figsize=(16, 7))
    for i, csv_f in enumerate(csv_fs):
        all_data = pd.read_csv(csv_f).replace(-100000, -1.1)
        data = all_data.loc[:9]
        data = data.drop(columns="Unnamed: 0")
        data = data.rename(columns={old_col: old_col[:20] + "-" + str(i) for i, old_col in enumerate(data.columns)})
        sns.boxplot(palette=sns.color_palette("hls"), data=data, ax=subs[i])
        subs[i].set_xticklabels(labels=data.columns, rotation=45)
        subs[i].title.set_text(csv_f.split("_")[0])
    fig.text(0.5, -0.15, 'Hyperparameters Options', ha='center')
    fig.text(0.04, 0.5, 'MI Score', va='center', rotation='vertical')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_elbow(data, save_path):
    sns.lineplot(x='x', y='y', palette=sns.color_palette("hls", 10), data=data,
                 legend="full")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# https://plotly.com/python/v3/ipython-notebooks/baltimore-vital-signs/ - plotly plots
# https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python
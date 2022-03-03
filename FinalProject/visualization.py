from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
filled_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


def plot_clusters_real_vs_predictions(data, targets, clustering_results, save_path):
    df = pd.DataFrame()
    df['x'] = data.values[:, 0]
    df['y'] = data.values[:, 1]
    df["clustering_results"] = clustering_results
    num_targets = targets.shape[1]
    targets.index = df.index
    fig, subs = plt.subplots(nrows=1, ncols=num_targets, figsize=(16, 7))
    for i in range(num_targets):
        col_name = targets.columns[i]
        df['target'] = targets[col_name]
        sub_df = df.sample(1000, random_state=0)
        sns.scatterplot(x='x', y='y', hue='clustering_results',
                        palette=sns.color_palette("hls", len(np.unique(sub_df['clustering_results']))),
                        data=sub_df,
                        style="target",
                        markers=filled_markers,
                        ax=subs[i])  # legend="full",
        subs[i].title.set_text(col_name)
        subs[i].legend(ncol=10, loc="lower center")
    # plt.legend(ncol=10, loc="lower center")
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
    algs = sorted(set([x.split("_")[0] for x in data.columns if "cmp" in x]))
    df = pd.DataFrame()
    df["clustering_results"] = data["clustering_results"]
    df["iYearwrk"] = data["iYearwrk"]
    fig, subs = plt.subplots(nrows=len(algs), ncols=2, figsize=(5, 2.5 * len(algs)), sharey="row")
    for i, dim_reduc_alg in enumerate(algs):
        df['cmp 1'] = data[f"{dim_reduc_alg}_cmp1"]
        df['cmp 2'] = data[f"{dim_reduc_alg}_cmp2"]
        sub_df = df #.sample(1000, random_state=0)
        for j, hue in enumerate(["iYearwrk", "clustering_results"]):
            colors = sns.color_palette("hls", len(np.unique(sub_df[hue])))
            if hue == "clustering_results" and -1 in sub_df[hue].values:
                colors.insert(0, (0, 0, 0))
                colors = colors[:-1]
            sns.scatterplot(x='cmp 1', y='cmp 2', hue=hue,
                            palette=colors, data=sub_df,  ax=subs[i, j]) #legend="full",
            subs[i, j].tick_params(axis='both', which='both', labelsize=8)
            title = f"{dim_reduc_alg}"
            if i==0:
                title = f"{hue}\n{dim_reduc_alg}"
            subs[i, j].set_title(title, size=12, x=0.5, y=0.85)
            subs[i, j].set_xlabel("")
            subs[i, j].set_ylabel("")
            subs[i, j].legend([])#box_to_anchor=bbox_to_anchor)  # , loc="center left",)
            # subs[i, j].set_aspect('equal')
    fig.text(0.5, 0.04, "principle component 1", ha="center", va="center")
    fig.text(0.05, 0.5, "principle component 2", ha="center", va="center", rotation=90)
    handles, labels = subs[0,0].get_legend_handles_labels()
    legend1 = plt.legend(handles, labels,  bbox_to_anchor=(-1.5,5))#, loc='center')
    handles, labels = subs[0,1].get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(2,5))#, loc='center')
    plt.gca().add_artist(legend1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_combined_boxplots(csv_fs, save_path, ylabel="MI score", xlabel="Hyperparameters Options", ylim=(-0.02,1)):
    """
    plot the MIscore distribution using boxplot
    :param csv_fs: list of all the csv files we want to plot theirs boxplot
    :param save_path: where to save the plot
    :return: None
    """
    combined_df = pd.DataFrame()
    for i, csv_f in enumerate(csv_fs):
        title = csv_f.stem.split("_")[0]
        csv_f = str(csv_f)
        all_data = pd.read_csv(csv_f).drop(columns="Unnamed: 0")
        data = all_data.loc[:9]
        values = []
        groups = []
        for col in data.columns:
            values.extend(data[col].values)
            groups.extend([col.split("=")[-1] for i in range(len(data[col]))])
        combined_df["ExternalVars"] = groups
        combined_df[title] = values
    value_vars = [col_n for col_n in combined_df.columns if col_n!='ExternalVars']
    dd = pd.melt(combined_df, id_vars=['ExternalVars'],
                 value_vars=value_vars, var_name='Clustering Algs', value_name="MI Score")
    dd["MI Score"] = dd["MI Score"].astype(float)
    ax = sns.boxplot(x='ExternalVars', y='MI Score', data=dd, hue='Clustering Algs', color='Clustering Algs', palette=sns.color_palette("hls"))
    boxes = ax.artists
    for i, box in enumerate(boxes):
        box.set_facecolor(sns.color_palette("hls")[i%3])
        box.set_edgecolor(sns.color_palette("hls")[i%3])

    plt.ylabel("MI Score", fontsize=15)
    plt.xlabel("ExternalVars", fontsize=15)
    plt.savefig(save_path.replace('png','svg'), dpi=300,bbox_inches='tight')


def plot_boxplots(csv_fs, save_path, ylabel="MI score", xlabel="Hyperparameters Options", ylim=(0,1)):
    """
    plot the some score (MI or Silhouette) distribution using boxplot
    :param csv_fs: list of all the csv files we want to plot theirs boxplot
    :param save_path: where to save the plot
    :return: None
    """
    fig, subs = plt.subplots(1, len(csv_fs), figsize=(15, 5))
    for i, csv_f in enumerate(csv_fs):
        title = csv_f.stem.split("_")[0]
        csv_f = str(csv_f)
        all_data = pd.read_csv(csv_f).replace(-100000, -1.1)
        data = all_data.loc[:9]
        data = data.drop(columns="Unnamed: 0")
        data = data.rename(columns={old_col: (old_col.split("=")[-1]) for i, old_col in enumerate(data.columns)})
        sns.boxplot(palette=sns.color_palette("hls"), data=data, ax=subs[i])
        subs[i].set_xticklabels(labels=data.columns, rotation=-45, fontsize=12)
        subs[i].set_yticklabels(labels=np.arange(ylim[0], ylim[1], 0.02), fontsize=12)
        subs[i].title.set_text(title)
        subs[i].title.set_size(18)
        subs[i].set_ylim(ylim)
    fig.text(0.5, -0.1, xlabel, ha='center', size=18)
    fig.text(0.05, 0.5, ylabel, va='center', rotation='vertical', size=18)
    # plt.show()
    plt.savefig(save_path,dpi=300,bbox_inches='tight')


def plot_all_boxplots(dir_with_csvs, combined=False):
    save_path = "results/clustering_evaluation/{}.png"
    external_vars_csvs = [file for file in dir_with_csvs.iterdir() if file.suffix==".csv" and "external_vars" in file.stem]
    hyperparams_csvs = [file for file in dir_with_csvs.iterdir() if file.suffix==".csv" and "hyperparam" in file.stem]
    if combined:
        plot_combined_boxplots(external_vars_csvs, save_path.format("external_vars_cmp"), ylabel="MI score", xlabel="External Variables")
        plot_combined_boxplots(hyperparams_csvs, save_path.format("hyperparam_cmp"), ylabel="Silhouette score", xlabel="Hyperparameters Options", ylim=(-1,1))
    else:
        plot_boxplots(external_vars_csvs, save_path.format("external_vars_cmp"), ylabel="MI score",
                      xlabel="External Variables")
        plot_boxplots(hyperparams_csvs, save_path.format("hyperparam_cmp"), ylabel="Silhouette score",
                      xlabel="Hyperparameters Options", ylim=(-1, 1))


def plot_elbow(data, save_path):
    sns.lineplot(x='x', y='y', palette=sns.color_palette("hls", 10), data=data,
                 legend="full")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# https://plotly.com/python/v3/ipython-notebooks/baltimore-vital-signs/ - plotly plots
# https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python

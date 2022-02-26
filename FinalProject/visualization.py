import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_clusters(data, targets, save_path):
    df = pd.DataFrame()
    df['x'] = data.values[0]
    df['y'] = data.values[1]
    num_targets = targets.shape[0]
    fig, subs = plt.subplots(nrows=1, ncols=num_targets, figsize=(16, 7))
    for i in range(num_targets):
        df['target'] = targets[i]
        sns.scatterplot(x='x', y='y', hue='target', palette=sns.color_palette("hls", 10), data=df,
                        legend="full", ax=subs[i])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_silhouette_boxplots(csv_fs, save_path):
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
        sns.boxplot(palette=sns.color_palette("Oranges"), data=data, ax=subs[i])
        subs[i].set_xticklabels(labels=data.columns, rotation=45)
        subs[i].title.set_text(csv_f.split("_")[0])
    fig.text(0.5, -0.15, 'Hyperparameters Options', ha='center')
    fig.text(0.04, 0.5, 'Silhouette Score', va='center', rotation='vertical')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_elbow(data, save_path):
    sns.lineplot(x='x', y='y', palette=sns.color_palette("hls", 10), data=data,
                        legend="full")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# https://plotly.com/python/v3/ipython-notebooks/baltimore-vital-signs/ - plotly plots
# https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python

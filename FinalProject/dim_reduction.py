import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import prince
algo_types_dim_reduction_params = {
        "PCA": {},
        "CMDS": {},
        "TSNE": {},
        "ISO": {"n_neighbors": [50]},
        "LLE": {"n_neighbors": [50]},
        "EigenMaps": {"n_neighbors": [50]}
}
dim_reduction_algs = [alg_n for alg_n in algo_types_dim_reduction_params]


def find_topmost_describable_features_in_mca(model):
    normed_ev = np.array(model.eigenvalues_) / sum(model.eigenvalues_)
    cum_eigenvalues_ = np.cumsum(normed_ev)
    plt.plot(range(len(cum_eigenvalues_)), cum_eigenvalues_)
    num_of_dims = np.where((cum_eigenvalues_ > 0.9) & (cum_eigenvalues_ < 0.905))
    plt.plot(num_of_dims[0] * [1, 1], np.arange(0, 2, 1), color='r')
    plt.annotate(f"num of components \nthat explains 90% \nof the data={num_of_dims[0]}", xy=(num_of_dims[0], 0.9))
    n_cmpnt = 20
    plt.plot([n_cmpnt, n_cmpnt], np.arange(0, 2, 1), color='r')
    plt.annotate(f"explainability perc \nat {n_cmpnt} components={round(cum_eigenvalues_[n_cmpnt], 2)}",
                 xy=(n_cmpnt, cum_eigenvalues_[n_cmpnt]))
    plt.bar(range(len(normed_ev)), normed_ev, color='b')
    plt.ylabel("eigenvalues")
    plt.xlabel("principle components")
    # [(explained_inertia_>90) & (explained_inertia_<95)], range(100))
    plt.savefig("plots/MCA_scree_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def preprocess_dim_reduction(data, n_components=20):
    mca = prince.MCA(
        n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )
    return mca.fit_transform(data)
    #
    # ax = model.plot_coordinates(
    #     X=data,
    #     ax=None,
    #     figsize=(8, 10),
    #     show_row_points=False,
    #     row_points_size=0,
    #     show_row_labels=False,
    #     show_column_points=True,
    #     column_points_size=30,
    #     show_column_labels=True,
    #     legend_n_cols=1
    # ).legend(loc='center left', bbox_to_anchor=(1, 0.5))


def apply_dim_reduction(x, alg_type, hyper_params_config, n_components=2):
    if alg_type == "PCA":
        x = StandardScaler().fit_transform(x)  # normalizing the features
        model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "CMDS":
        model = MDS(n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "TSNE":
        model = TSNE(n_components=n_components, learning_rate='auto',metric="hamming")
        # score = model.kl_divergence_
        transformed_data = model.fit_transform(x)
    elif alg_type == "ISO":
        model = Isomap(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "LLE":
        model = LocallyLinearEmbedding(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    elif alg_type == "EigenMaps":
        model = SpectralEmbedding(n_neighbors=hyper_params_config["n_neighbors"], n_components=n_components)
        transformed_data = model.fit_transform(x)
    else:
        raise Exception("unrecognized algorithm type")

    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1', 'principle_cmp2'])

    return transformed_data_df, model


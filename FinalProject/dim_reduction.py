import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
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


dim_reduction_algs = ["PCA", "CMDS", "ISO", "LLE", "EigenMaps"]


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
        score = model.kl_divergence_
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


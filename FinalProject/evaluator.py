import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from tqdm import tqdm

from anomaly_detection import anomaly_detect_algs, apply_anomaly_detection, algo_types_anomaly_params
from dim_reduction import dim_reduction_algs, apply_dim_reduction, algo_types_dim_reduction_params
from clustering import algo_types_clustering_params, clustering_algs, apply_clustering
from statistic_tests import StatTester
import joblib

# TODO: move later to globals
stats_dir = "./stats"
external_variables = ["dAge", "dHispanic", "iYearwrk", "iSex"]
DIM_REDUCTION= "dim_reduction"

class Evaluator:
    def __init__(self, data, transformed_data=None, num_of_iterations_for_statistical_analysis=10,
                 eval_on_test=False, algs_type="clustering", do_one_hot=False,save_path="", model_path=None):
        self.data = data
        self.transformed_data = transformed_data
        self.data_subsets = None
        self.external_tags = None
        self.all_hyper_params_results = {}
        self.subset_fraction=0.5
        self.num_of_iterations_for_statistical_analysis = num_of_iterations_for_statistical_analysis
        self.eval_on_test = eval_on_test
        self.algs_type = algs_type
        self.model_counter = 0
        self.scores_to_extract = ["silhouette", "mi"]
        self.do_one_hot = do_one_hot
        if self.algs_type == "clustering":
            self.algs = clustering_algs
            self.params_config = algo_types_clustering_params
        elif algs_type == DIM_REDUCTION:
            self.algs = dim_reduction_algs
            self.params_config = algo_types_dim_reduction_params
            self.OH_encoder = OneHotEncoder(sparse=False).fit(self.data.drop(columns=external_variables))
            self.num_of_iterations_for_statistical_analysis = 1
            self.subset_fraction = 0.01
            self.scores_to_extract = ["vis"]
            if model_path is None:
                model_path = self.load_model("results/MODELS/clustering/KModes_2_3.model")
            # self.clustering_model = self.load_model("results/MODELS/clustering/DBSCAN_0.05_2_'hamming'_5.model")
            self.clustering_model = self.load_model(model_path)
            self.reduction_results = pd.DataFrame()

        elif algs_type == "anomaly_detect":
            self.algs = anomaly_detect_algs
            self.params_config = algo_types_anomaly_params
        else:
            raise Exception("unknown algs type")
        self.statistic_tester = StatTester()
        self.save_all_models = False

    def run_single_algo(self, data, alg_type, hyper_params_config):
        if alg_type in clustering_algs:
            results, model = apply_clustering(data, alg_type, hyper_params_config)
        elif alg_type in dim_reduction_algs:
            results, model = apply_dim_reduction(data, alg_type, hyper_params_config)
        elif alg_type in anomaly_detect_algs:
            results, model = apply_anomaly_detection(data, alg_type, hyper_params_config)
        else:
            raise Exception("unKnown alg type")

        if model is not None and self.save_all_models:
            model_file_name = "./MODELS/{0}_{1}_{2}.model".format(
                alg_type, str(hyper_params_config), self.model_counter)
            joblib.dump(model, model_file_name)  # save model to file
            self.model_counter += 1

        return results, model

    def get_subsets(self, full_data, num_subsets):
        subsets = []
        tags = []
        for test_iteration in range(num_subsets):
            part_data = full_data.sample(frac=self.subset_fraction, random_state=test_iteration)
            train_subset, test_subset = train_test_split(part_data, test_size=0.2,
                                                         random_state=test_iteration)  # choose data for this iteration
            tags.append((train_subset[external_variables], test_subset[external_variables]))
            drop_cols = external_variables
            train_subset.drop(columns=drop_cols, inplace=True)
            test_subset.drop(columns=drop_cols, inplace=True)
            if self.do_one_hot and self.algs_type == DIM_REDUCTION:
                train_subset = pd.DataFrame(self.OH_encoder.transform(train_subset), index=train_subset.index)
                test_subset = pd.DataFrame(self.OH_encoder.transform(test_subset), index=test_subset.index)
                indxs = train_subset.index.union(test_subset.index).astype(int)
                if self.transformed_data is not None:
                    self.transformed_data = self.transformed_data.loc[indxs].values
            subsets.append((train_subset, test_subset))
        return subsets, tags

    def apply_algs_with_all_hyper_params_configs(self):
        self.data_subsets, self.external_tags = self.get_subsets(self.data,
                                                                 self.num_of_iterations_for_statistical_analysis)
        for alg_name in tqdm(self.algs):
            self.all_hyper_params_results[alg_name] = {}
            hyper_params_config = self.params_config[alg_name]
            hyper_params_config_keys = list(hyper_params_config.keys())
            hyper_parameters_config_options = list(itertools.product(*hyper_params_config.values()))
            current_alg_hyper_params_results = self.run_all_hyper_parameters_combos(alg_name, hyper_params_config_keys,
                                                                                    hyper_parameters_config_options)
            self.all_hyper_params_results[alg_name] = current_alg_hyper_params_results

    def run_all_hyper_parameters_combos(self, alg, hyper_params_config_keys, hyper_parameters_config_options):
        current_alg_hyper_params_results = {}
        for hyper_parameters_config in hyper_parameters_config_options:
            combo_config = self.convert_tuple2config(hyper_params_config_keys, hyper_parameters_config)
            scores = self.extract_scores(alg, combo_config, scores_to_extract=self.scores_to_extract)
            current_alg_hyper_params_results[tuple(hyper_parameters_config)] = scores
        return current_alg_hyper_params_results

    @staticmethod
    def calculate_score(data1, data2, score_method="silhouette", model=None):
        if score_method == "silhouette":
            if len(np.unique(data2)) > 1:
                return silhouette_score(data1, data2)
            return -100000
        elif score_method == "silhouette_per_sample":
            if len(np.unique(data2)) > 1:
                return silhouette_samples(data1, data2)
            return np.full(data2.shape, -100000)
        elif score_method == "mi":
            return mutual_info_score(data1, data2)
        elif "ic" in score_method and model is not None:
            try:
                if score_method == "aic":
                    return model.aic(data1)
                else:
                    return model.bic(data1)
            except Exception as e:
                print(e)
                return -100000
        else:
            raise Exception("unknown score method")

    def extract_scores(self, alg, combo_config, scores_to_extract=[]):
        scores = {score_m: [] for score_m in scores_to_extract}
        for ((train_d, test_d), (train_tags, test_tags)) in tqdm(zip(self.data_subsets, self.external_tags)):
            x = train_d.values if self.eval_on_test else pd.concat([train_d, test_d]).values
            y = train_tags if self.eval_on_test else pd.concat([train_tags, test_tags])
            n_labels, model = self.run_single_algo(x, alg, combo_config)
            labels = model.predict(test_d) if self.eval_on_test else n_labels
            eval_x = test_d if self.eval_on_test else x
            for score_method in scores_to_extract:
                if self.algs_type == DIM_REDUCTION:
                    self.reduction_results[f"{alg}_cmp1"] = labels['principle_cmp1']
                    self.reduction_results[f"{alg}_cmp2"] = labels['principle_cmp2']
                    if "clustering_results" not in self.reduction_results.columns:
                        self.reduction_results["iYearwrk"] = y['iYearwrk'].values
                        if self.transformed_data is None:
                            self.transformed_data = x
                        try:
                            self.reduction_results["clustering_results"] = self.clustering_model.predict(self.transformed_data)
                        except:
                            self.reduction_results["clustering_results"] = self.clustering_model.fit_predict(self.transformed_data)
                        #tags = test_tags if self.eval_on_test else pd.concat([train_tags, test_tags])
                        # plot_clusters(labels, tags, self.reduction_results["clustering_results"],f"{alg}.png")
                elif "silhouette" in score_method:
                    scores[score_method].append(self.calculate_score(eval_x, labels, score_method))
                elif "ic" in score_method:
                    scores[score_method].append(self.calculate_score(eval_x, labels, score_method, model=model))
                else:
                    tags = test_tags if self.eval_on_test else pd.concat([train_tags, test_tags])
                    out = scores.pop(score_method, None)
                    if out is not None:
                        scores.update({"external_vars": {col_n: {score_method: []} for col_n in tags.columns}})
                    for col_name in tags.columns:
                        scores["external_vars"][col_name][score_method].append(
                            self.calculate_score(labels, tags[col_name],
                                                 score_method))
        return scores

    def choose_best_hyperparams_config(self, alg_name, score_method="s_score"):
        alg_results = self.all_hyper_params_results[alg_name]
        hyper_parameters_names = str(list(self.params_config[alg_name]))
        save_path = f"{stats_dir}/{alg_name}_hyperparam.csv"
        best_key = self.statistic_tester.find_best_config_based_on_statistic_test(alg_results,
                                                                                  hyper_parameters_names,
                                                                                  quantitative_score=score_method,
                                                                                  save_path=save_path)
        print(f"alg type: {alg_name}, hyper param: {best_key}")
        return alg_results[best_key], best_key

    def choose_best_external_variable(self, alg_name, score_method="mi_score"):
        alg_results = self.all_hyper_params_results[alg_name]["external_vars"]
        hyper_parameters_names = "external_var"
        save_path = f"{stats_dir}/{alg_name}_external_vars_cmp.csv"
        best_key = self.statistic_tester.find_best_config_based_on_statistic_test(alg_results,
                                                                                  hyper_parameters_names,
                                                                                  quantitative_score=score_method,
                                                                                  save_path=save_path)
        print(f"alg type: {alg_name}, external variable: {best_key}")
        return alg_results[best_key], best_key

    def choose_best_alg(self, score_method="mi_score"):
        save_path = f"{stats_dir}/{self.algs_type}_cmp.csv"
        best_key = self.statistic_tester.find_best_config_based_on_statistic_test(self.all_hyper_params_results, None,
                                                                                  quantitative_score=score_method,
                                                                                  save_path=save_path)
        print(f"best alg: {best_key}")
        return self.all_hyper_params_results[best_key], best_key

    def algs_evaluation_pipeline(self, save_best=False):
        """
        :param save_best: whether to save best models and theirs results or not
        :return: None
        """
        self.apply_algs_with_all_hyper_params_configs()
        best_hps = {}
        if self.algs_type == DIM_REDUCTION:
            from visualization import plot_all_dim_reduc
            plot_all_dim_reduc(self.reduction_results, f"results/dim_reduction_cmp.png")
        else:
            for alg_name in self.algs:
                results, hp_best_key = self.choose_best_hyperparams_config(alg_name, "silhouette")
                best_hps[alg_name] = hp_best_key
                self.all_hyper_params_results[alg_name] = results
                results, ev_best_key = self.choose_best_external_variable(alg_name, "mi")
                self.all_hyper_params_results[alg_name] = results
            results, best_alg = self.choose_best_alg("mi")
            if save_best:
                return self.save_best_models_and_results(best_alg, best_hps[best_alg])
        return None

    def save_best_models_and_results(self, alg, best_params):
        results = pd.DataFrame()
        best_params_config = self.convert_tuple2config(list(self.params_config[alg].keys()),
                                                       best_params)
        best_params = str(list(best_params)).replace(", ", "_").replace("[", "").replace("]", "")
        base_models_p = f"results/MODELS/{self.algs_type}/{alg}_{best_params}"
        base_results_p = f"results/LABELS/{self.algs_type}"
        os.makedirs("/".join(base_models_p.split("/")[:-1]), exist_ok=True)
        os.makedirs(base_results_p, exist_ok=True)
        for subset_idx, (train_d, test_d) in enumerate(self.data_subsets):
            x = train_d if self.eval_on_test else pd.concat([train_d, test_d])
            labels, model = self.run_single_algo(x.values, alg_type=alg, hyper_params_config=best_params_config)
            model_file_name = f"{base_models_p}_{subset_idx}.model"
            joblib.dump(model, model_file_name)  # save model to file
            col_n = f"subset_{subset_idx}"
            results[col_n] = x.index
            results[f"{col_n}_labels"] = labels
        results.to_csv(f"{base_results_p}/{alg}_{best_params}.csv")
        return model_file_name

    def load_model(self, model_path):
        # best_params = str(list(best_params)).replace(", ", "_").replace("[", "").replace("]", "")
        model_file_name = f"{model_path}"
        return joblib.load(model_file_name)

    def convert_str2config(self, alg, hp_best_key):
        params_config = algo_types_clustering_params[alg].copy()
        params_names, params_values = hp_best_key.split("=")
        params_names, params_values = list(params_names), list(params_values)
        for param_n, param_v in zip(params_names, params_values):
            params_config[param_n] = param_v
        return params_config

    @staticmethod
    def convert_tuple2config(keys, values):
        config = {}
        for config_key_index in range(len(keys)):
            v = values[config_key_index]
            try:
                tmp_v = int(v)
                if v == tmp_v:
                    v=tmp_v
                else:
                    v= float(v)
            except:
                try:
                    v = float(v)
                except:
                    pass
            config[keys[config_key_index]] = v
        return config

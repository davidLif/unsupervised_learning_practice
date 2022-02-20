import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score
from tqdm import tqdm

from anomaly_detection import anomaly_detect_algs, apply_anomaly_detection, algo_types_anomaly_params
from dim_reduction import dim_reduction_algs, apply_dim_reduction
from clustering import algo_types_clustering_params, clustering_algs, apply_clustering
from statistic_tests import StatTester

# TODO: move later to globals
stats_dir = "./stats"
external_variables = ["dAge", "dHispanic", "iYearwrk", "iSex"]


class Evaluator:
    def __init__(self, data, num_of_iterations_for_statistical_analysis=10, eval_on_test=False, algs_type="clustering"):
        self.data = data
        self.data_subsets = None
        self.external_tags = None
        self.all_hyper_params_results = {}
        self.num_of_iterations_for_statistical_analysis = num_of_iterations_for_statistical_analysis
        self.eval_on_test = eval_on_test
        self.algs_type = algs_type
        if self.algs_type == "clustering":
            self.algs = clustering_algs
            self.params_config = algo_types_clustering_params
        elif algs_type == "dim_reduction":
            self.algs = dim_reduction_algs
            self.params_config = {}
        elif algs_type == "anomaly_detect":
            self.algs = anomaly_detect_algs
            self.params_config = algo_types_anomaly_params
        else:
            raise Exception("unknown algs type")
        self.statistic_tester = StatTester()

    def run_single_algo(self, data, alg_type, hyper_params_config):
        if alg_type in clustering_algs:
            results, model = apply_clustering(data, alg_type, hyper_params_config)
        elif alg_type in dim_reduction_algs:
            results, model = apply_dim_reduction(data, alg_type, hyper_params_config)
        elif alg_type in anomaly_detect_algs:
            results, model = apply_anomaly_detection(data, alg_type, hyper_params_config)
        else:
            raise Exception("unKnown alg type")

        return results, model

    def get_subsets(self, full_data, num_subsets):
        subsets = []
        tags = []
        for test_iteration in range(num_subsets):
            part_data = full_data.sample(frac=0.1, random_state=test_iteration)
            train_subset, test_subset = train_test_split(part_data, test_size=0.2,
                                                         random_state=test_iteration)  # choose data for this iteration
            tags.append((train_subset[external_variables], test_subset[external_variables]))
            drop_cols = external_variables
            subsets.append((train_subset.drop(columns=drop_cols), test_subset.drop(columns=drop_cols)))
        return subsets, tags

    def apply_algs_with_all_hyper_params_configs(self):
        self.data_subsets, self.external_tags = self.get_subsets(self.data, self.num_of_iterations_for_statistical_analysis)
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
            combo_config = {}
            for config_key_index in range(len(hyper_params_config_keys)):
                combo_config[hyper_params_config_keys[config_key_index]] = hyper_parameters_config[config_key_index]
            scores = self.extract_scores(alg, combo_config, scores_to_extract=["silhouette", "mi"])
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
            n_labels, model = self.run_single_algo(x, alg, combo_config)
            labels = model.predict(test_d) if self.eval_on_test else n_labels
            eval_x = test_d if self.eval_on_test else x
            for score_method in scores_to_extract:
                if "silhouette" in score_method:
                    scores[score_method].append(self.calculate_score(eval_x, labels, score_method))
                elif "ic" in score_method:
                    scores[score_method].append(self.calculate_score(eval_x, labels, score_method, model=model))
                else:
                    tags = test_tags if self.eval_on_test else pd.concat([train_tags, test_tags])
                    out = scores.pop(score_method, None)
                    if out is not None:
                        scores.update({"external_vars": {col_n: {score_method: []} for col_n in tags.columns}})
                    for col_name in tags.columns:
                        scores["external_vars"][col_name][score_method].append(self.calculate_score(labels, tags[col_name],
                                                                                   score_method))
        return scores

    def choose_best_hyperparams_config(self, alg_name, score_method="s_score"):
        alg_results = self.all_hyper_params_results[alg_name]
        hyper_parameters_names = str(list(alg_results))
        save_path = f"{stats_dir}/{alg_name}_hyperparam.csv"
        best_key = self.statistic_tester.find_best_config_based_on_statistic_test(alg_results,
                                                                                  hyper_parameters_names,
                                                                                  quantitative_score=score_method,
                                                                                  save_path=save_path)
        print(f"alg type: {alg_name}, hyper param: {best_key}")
        return alg_results[best_key], best_key

    def choose_best_external_variable(self, alg_name, score_method="mi_score"):
        alg_results = self.all_hyper_params_results[alg_name]["external_vars"]
        hyper_parameters_names = str(list(alg_results))
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

    def algs_evaluation_pipeline(self):
        """

        :return:
        """
        self.apply_algs_with_all_hyper_params_configs()

        for alg_name in self.algs:
            results, best_key = self.choose_best_hyperparams_config(alg_name, "silhouette")
            self.all_hyper_params_results[alg_name] = results
            results, best_key = self.choose_best_external_variable(alg_name, "mi")
            self.all_hyper_params_results[alg_name] = results
        self.choose_best_alg("mi")

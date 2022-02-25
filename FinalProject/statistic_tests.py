import numpy as np
import pandas as pd
import csv
from scipy import stats


def extract_results_per_subsets(data, labels, algs_list, quantitative_test):
    results = {}
    subsets, parted_labels = [data], [labels]  # no real subsets for now
    for alg in algs_list:
        alg_res = []
        for subset, sub_lbls in zip(subsets, parted_labels):
            pred_labels = alg.fit_predict(subset)
            alg_res.append(quantitative_test(sub_lbls, pred_labels))
        results[alg.__name__] = alg_res
    return results


def print_stat_quant_results(stats, pvalue, algs_list, quant_list):
    to_print = f"stats: {stats},\tpvalue: {pvalue}"
    for alg_n, quants in zip(algs_list, quant_list):
        to_print += f",\t{alg_n} mean quant: {np.mean(quants)}"
    to_print += "\n"
    print(to_print)


def apply_paired_ttest(data, labels, alg1, alg2, quantitative_test):
    alg1_n = alg1.__name__
    alg2_n = alg2.__name__
    results = extract_results_per_subsets(data, labels, [alg1, alg2], quantitative_test)
    res1, res2 = results[alg1_n], results[alg2_n]
    t_stat, pvalue = stats.ttest_rel(res1, res2)
    print_stat_quant_results(t_stat, pvalue, [alg1_n, alg2_n], [res1, res2])


def find_topk(all_res, k):
    best_idxs = np.argsort(list(map(lambda x: np.mean([-100000 if item is None else item for item in x]), all_res)))[
                ::-1][:k]
    best_res = np.array(all_res)[best_idxs]
    return best_res, best_idxs


class StatTester:
    def __init__(self):
        self.params_keys = []
        self.data, self.means_list, self.anova_pvalue, self.t_pvalue, self.top2_idxs = None, None, None, None, None

    def find_best_config_based_on_statistic_test(self, quantitative_data, hyper_parameters_names
                                                 , quantitative_score="s_score", save_path=""):
        self.apply_statistic_test(quantitative_data, quantitative_score)
        self.save_results_to_csv(hyper_parameters_names, save_path)
        return self.stat_test_results_and_update_csv(quantitative_score, save_path)

    def apply_statistic_test(self, quantitative_data, quantitative_score="silhouette"):
        self.params_keys = list(quantitative_data.keys())

        comparing_scores = [quantitative_data[key][quantitative_score] for key in self.params_keys]
        num_data_to_compare = len(quantitative_data.keys())
        best_res = np.array(comparing_scores)
        anova_pvalue = -1
        try:

            if num_data_to_compare == 1:
                # stat, anova_pvalue = stats.f_oneway(best_res[0], best_res[1])
                best_res = np.vstack((best_res, best_res))
                pass
            elif num_data_to_compare == 2:
                # stat, anova_pvalue = stats.f_oneway(best_res[0], best_res[1])
                pass
            elif num_data_to_compare >= 3:
                stat, anova_pvalue = stats.f_oneway(*best_res)
            else:
                raise Exception(f"{num_data_to_compare}, this option doesnt exist")
            self.data = np.array(best_res).transpose()
            best_res, best_idxs = find_topk(best_res, 2)
            self.top2_idxs = best_idxs
            t_stat, self.t_pvalue = stats.ttest_rel(best_res[0], best_res[1])
        except Exception as e:
            print("Exception on t-test/anova")
            raise e
        self.means_list = [res.mean() for res in self.data.transpose()]
        self.anova_pvalue = anova_pvalue

    def save_results_to_csv(self, hyper_parameters_names, save_path=""):
        if hyper_parameters_names is None:
            columns_names = self.params_keys
        else:
            columns_names = [f"{hyper_parameters_names}={param_key}" for param_key in self.params_keys]

        df = pd.DataFrame(self.data, columns=columns_names)
        df.loc['mean'] = self.means_list
        df.to_csv(save_path)

    def stat_test_results_and_update_csv(self, quantitative_score, save_path):
        if self.anova_pvalue > 0.05:
            print("null hypothesis of anova accepted - cannot claim distinguishability. Will take the first option")
            with open(save_path, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
                writer.writerow("")
                writer.writerow(["Null hypothesis of anova accepted with pvalue", self.anova_pvalue])

            return self.params_keys[0]
        elif self.t_pvalue > 0.05:
            print("null hypothesis of paired ttest accepted - take first config")
            with open(save_path, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
                writer.writerow("")
                writer.writerow(["anova p-value score:", self.anova_pvalue])
                writer.writerow(["Null hypothesis of t-test accepted with pvalue", self.t_pvalue])

            return self.params_keys[0]
        else:
            if quantitative_score == "mi_score":
                quantitative_score_name = "mutual_info_score"
            elif quantitative_score == "s_score":
                quantitative_score_name = "silhouette_score"
            else:
                quantitative_score_name = quantitative_score

            max_mean_list = max(np.array(self.means_list)[self.top2_idxs])
            chosen_parm_key = self.params_keys[self.means_list.index(max_mean_list)]

            with open(save_path, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
                writer.writerow("")
                writer.writerow([f"Max mean {quantitative_score_name}:", max_mean_list])
                writer.writerow([f"Max mean Generated by:", chosen_parm_key])
                writer.writerow(["anova p-value score:", self.anova_pvalue])
                writer.writerow(["t-test p-value score:", self.t_pvalue])
            return chosen_parm_key

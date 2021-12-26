import numpy as np
import pandas as pd
import csv
# Statistics tests
# paired t-test
# assumptions:
# # dependent variable (DV) must be continuous V
# # observations are independent V
# # DV should be approximately normally distributed ?
# # # plot histogram / use shapiro test to check if the data is normally distributed
# # # (if this is violated najorly than we need to use other tests) - Wilcoxon signed-rank Test
# # DV should not contain any significant outliers - check with boxplot
# anova
# assumptions:
# # same as above
# # variances are equal between treatment groups (Levene’s or Bartlett’s Test)

from scipy import stats


# stat tests
def is_normally_distributed(data):
    w_value, pvalue = stats.shapiro(data)
    return pvalue > 0.05


def is_samples_from_populations_have_equal_variances(data_1, data_2, data_3):
    w, pvalue = stats.bartlett(data_1, data_2, data_3)
    return pvalue > 0.05


# the null hypothesis of p_ttest is that the diff between the variences of two variables is zero
def is_the_null_h_of_paired_ttest_is_rejected(data_1, data_2):
    ttest, pvalue = stats.ttest_rel(data_1, data_2)
    return pvalue < 0.05


# the null hypothesis of anova (f test) is that all group means are equal
def is_the_null_h_of_anova_test_is_rejected(data_1, data_2, data_3):
    Ftest, pvalue = stats.f_oneway(data_1, data_2, data_3)
    return pvalue < 0.05


def get_subsets(data, labels, n_subsets=50):
    # todo needs to be implemented
    # returns list of subsets and list of corresponded labels for each subset
    subsets, subsets_labels = [data], [labels]
    return subsets, subsets_labels


def extract_results_per_subsets(data, labels, algs_list, quantitative_test):
    results = {}
    subsets, parted_labels = get_subsets(data, labels)
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


def apply_anova_test(data, labels, alg1, alg2, alg3, quantitative_test):
    alg1_n = alg1.__name__
    alg2_n = alg2.__name__
    alg3_n = alg3.__name__
    results = extract_results_per_subsets(data, labels, [alg1, alg2, alg3], quantitative_test)
    res1, res2, res3 = results[alg1_n], results[alg2_n], results[alg3_n]
    f_stat, pvalue = stats.f_oneway(res1, res2, res3)
    print_stat_quant_results(f_stat, pvalue, [alg1_n, alg2_n, alg3_n], [res1, res2, res3])


def find_topk(all_res, k):
    best_idxs = np.argsort(list(map(lambda x: np.mean([-100000 if item is None else item for item in x]), all_res)))[::-1][:k]
    best_res = np.array(all_res)[best_idxs]
    return best_res, best_idxs


def find_best_config_based_on_statistic_test(quantitative_data, hyper_parameters_names
                                             , quantitative_score="s_score", save_path=""
                                             , best_k_to_save=2):
    params_keys = list(quantitative_data.keys())

    comparing_scores = [quantitative_data[key][quantitative_score] for key in params_keys]
    if len(quantitative_data.keys()) > 2:
        best_res, best_idxs = find_topk(comparing_scores, best_k_to_save)
        if best_k_to_save > 2:
            res1, res2, res3 = [-100000 if item is None else item for item in best_res[0]] \
                , [-100000 if item is None else item for item in best_res[1]]\
                , [-100000 if item is None else item for item in best_res[2]]
        else:
            res1, res2 = [-100000 if item is None else item for item in best_res[0]]\
                , [-100000 if item is None else item for item in best_res[1]]
        params_keys = list(np.array(params_keys)[best_idxs])
    else:
        res1, res2 = comparing_scores[0], comparing_scores[1]

    anova_failed = False
    anova_pvalue = -1
    try:
        if best_k_to_save > 2 and len(quantitative_data.keys()) > 2:
            stat, anova_pvalue = stats.f_oneway(res1, res2, res3)

            if anova_pvalue > 0.05:
                anova_failed = True

            data = np.array([res1, res2, res3]).transpose()
        else:
            data = np.array([res1, res2]).transpose()

        stat, pvalue = stats.ttest_rel(res1, res2)

    except Exception as e:
        print("Exception on t-test/anova")
        raise e

    if hyper_parameters_names is None:
        columns_names = params_keys
    else:
        columns_names = [f"{hyper_parameters_names}={param_key}" for param_key in params_keys]

    df = pd.DataFrame(data, columns=columns_names)
    df.to_csv(save_path)

    if anova_failed:
        print("null hypothesis of anova accepted - cannot claim distinguishability. Will take the first option")

        with open(save_path, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow("")
            writer.writerow(["Null hypothesis of anova accepted with pvalue", anova_pvalue])
    elif pvalue > 0.05:
        print("null hypothesis of paired ttest accepted - take first config")

        with open(save_path, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow("")
            writer.writerow(["Null hypothesis of t-test accepted with pvalue", pvalue])

        return params_keys[0]
    else:
        if quantitative_score == "mi_score":
            quantitative_score_name = "mutual_info_score"
        elif quantitative_score == "s_score":
            quantitative_score_name = "silhouette_score"
        else:
            quantitative_score_name = quantitative_score

        means_list = [np.mean(res1), np.mean(res2)]
        chosen_parm_key = params_keys[means_list.index(max(means_list))]

        with open(save_path, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow("")
            writer.writerow([f"Max mean {quantitative_score_name}:", stat])
            writer.writerow([f"Max mean Generated by:", chosen_parm_key])
            writer.writerow(["t-test p-value score:", pvalue])

        return chosen_parm_key

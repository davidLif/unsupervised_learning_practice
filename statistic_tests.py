import numpy as np
import pandas as pd
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
    best_idxs = np.argsort(list(map(lambda x: np.mean([-10 if item is None else item for item in x]), all_res)))[::-1][:k]
    best_res = np.array(all_res)[best_idxs]
    return best_res, best_idxs


def find_best_config_based_on_statistic_test(quantiative_data, quantitative_score="s_score", save_path="", use_anova=False):
    """

    :param quantiative_data:
    :param quantitative_score:
    :param save_path:
    :return: key of best config
    """
    params_keys = list(quantiative_data.keys())
    all_res = [quantiative_data[key][quantitative_score] for key in params_keys]
    if len(quantiative_data.keys()) > 2:
        k = 2
        if use_anova:
            k = 3
        best_res, best_idxs = find_topk(all_res, k)
        res1, res2 = [-10 if item is None else item for item in best_res[0]], [-10 if item is None else item for item in best_res[1]]
        params_keys = list(np.array(params_keys)[best_idxs])
    else:
        res1, res2 = all_res[0], all_res[1]
    try:
        if use_anova:
            res1, res2, res3 = best_res[0], best_res[1], best_res[2]
            stat, pvalue = stats.f_oneway(res1, res2, res3)
            data = np.array([res1, res2, res3]).transpose()
        else:
            stat, pvalue = stats.ttest_rel(res1, res2)
            data = np.array([res1, res2]).transpose()
    except:
        print()
    df = pd.DataFrame(data, columns=params_keys)
    df.loc["stats score"] = stat
    df.loc["p value"] = pvalue
    df.to_csv(save_path)
    if pvalue > 0.05:
        print("null hypothesis of paired ttest accepted - take first config")
        return params_keys[0]
    else:
        mean1, mean2 = np.mean(res1), np.mean(res2)
        if mean1 > mean2:
            return params_keys[0]
        return params_keys[1]
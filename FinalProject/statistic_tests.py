from scipy import stats


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


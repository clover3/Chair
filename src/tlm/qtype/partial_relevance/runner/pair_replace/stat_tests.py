import numpy as np
import scipy.stats

from cache import load_from_pickle
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import IDX_FUNC, IDX_CONTENT


def main():
    score_d1 = load_from_pickle("ps_replace_rates_100words")
    score_d2 = load_from_pickle("ps_replace_rates_empty")
    n_sample = 10
    for option_as_metric in ["recall", "precision"]:
        for target_idx in [IDX_FUNC, IDX_CONTENT]:
            common_key = option_as_metric, target_idx
            print(common_key)
            for source_idx, score_d in enumerate([score_d1, score_d2]):
                key_good = option_as_metric, False, target_idx
                key_bad = option_as_metric, True, target_idx
                scores_good = score_d[key_good][:n_sample]
                scores_bad = score_d[key_bad][:n_sample]

                diff = np.array([s1-s2 for s1, s2 in zip(scores_good, scores_bad)])
                data = diff,
                # res = scipy.stats.bootstrap(data, np.std, )
                avg_diff, p_value = scipy.stats.ttest_rel(scores_good, scores_bad)
                print(source_idx, avg_diff, p_value)


if __name__ == "__main__":
    main()
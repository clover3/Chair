from scipy.stats import ttest_rel

from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join
from tab_print import print_table


def main():
    sota_retrieval_methods = [
        "ce_msmarco_mini_lm",
        "splade",
        "tas_b",
        "contriever",
        "contriever-msmarco",
    ]
    for method_name in sota_retrieval_methods:
        print(method_name)
        save_path = path_join(output_path, "mmp", "append_test", method_name)
        d = load_pickle_from(save_path)

        base = d['a']
        table = []
        for k, target in d.items():
            avg_gap, p_value = ttest_rel(base, target)
            row = [k, avg_gap, p_value, p_value < 0.01]
            table.append(row)
        print_table(table)

    return NotImplemented


if __name__ == "__main__":
    main()
from collections import Counter

from table_lib import tsv_iter
from list_lib import index_by_fn
from misc_lib import group_by, get_first
from tab_print import tab_print_dict
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import term_align_candidate2_score_path, \
    mmp_root
from cpath import output_path
from misc_lib import path_join


def categorize_one(score):
    if score > 0.01:
        return "POS"
    elif score < -0.01:
        return "NEG"
    else:
        return "ZERO"


def categorize_pair(train_score, dev_score):
    c1 = categorize_one(train_score)
    c2 = categorize_one(dev_score)
    if c1 == c2 and c1 != "ZERO":
        return "Correct"
    elif c1 == "ZERO" or c2 == "ZERO":
        return "Undecided"
    else:
        return "Wrong"


def main():
    train_score_path = path_join(
        mmp_root(), "align_scores", "candidate2.tsv")
    dev_score_path = path_join(
        mmp_root(), "align_scores", "candidate2_dev.tsv")

    def get_key(triplet):
        q_term, d_term, score = triplet
        return q_term, d_term

    compare_counter = Counter()
    paired_counter = Counter()
    train_score_entry_d = index_by_fn(get_key, tsv_iter(train_score_path))
    for entry in tsv_iter(dev_score_path):
        key = get_key(entry)
        try:
            q_term, d_term, score = entry
            train_entry = train_score_entry_d[key]
            q_term_, d_term_, train_score_s = train_entry

            train_score = float(train_score_s)
            dev_score = float(score)

            category = categorize_pair(train_score, dev_score)
            compare_counter[category] += 1
            c1 = categorize_one(train_score)
            c2 = categorize_one(dev_score)
            paired_counter[c1, c2] += 1
        except KeyError:
            pass
    tab_print_dict(compare_counter)
    tab_print_dict(paired_counter)


    tp = paired_counter['POS', 'POS']
    fp = paired_counter['POS', 'NEG']

    tn = paired_counter['NEG', 'NEG']
    fn = paired_counter['NEG', 'POS']

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision', precision)
    print('recall', recall)


if __name__ == "__main__":
    main()

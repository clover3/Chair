import sys
from collections import Counter, defaultdict

from scipy.stats import ttest_rel

from cpath import output_path
from list_lib import right
from misc_lib import path_join, get_second, average
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import load_table
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list


def analyze_pairwise_ttest(score_log, term_list,
                           score_d: dict[str, float], t):

    entity_scores = defaultdict(list)
    for row in score_log:
        scores = list(map(float, row[2:]))
        for idx, s in enumerate(scores):
            entity_scores[term_list[idx]].append(scores[idx])


    sig_counter = Counter()
    for idx, term in enumerate(term_list):
        table_score1 = score_d[term]
        for idx2 in range(idx+1, len(term_list)):
            term2 = term_list[idx2]
            table_score2 = score_d[term2]
            if abs(table_score1-table_score2) > t:
                s1 = entity_scores[term]
                s2 = entity_scores[term2]
                avg_gap, p_value = ttest_rel(s1, s2)
                if p_value < 0.01:
                    sig_counter["yes"] += 1
                else:
                    sig_counter["no"] += 1
                # print(term, term2, avg_gap, p_value)

    return sig_counter["yes"], sig_counter["no"]


def analyze_pairwise_ttest_car(file_path, table_path):
    score_log = list(tsv_iter(file_path))
    term_list = load_car_maker_list()
    table = load_table(table_path)
    qterm = "car"
    for t in [0., 0.05, 0.1, 0.15]:
        ret = analyze_pairwise_ttest(score_log, term_list, table[qterm], t)
        print(t, ret)



def main():
    file_path = sys.argv[1]
    table_path = sys.argv[2]
    analyze_pairwise_ttest_car(file_path, table_path)



if __name__ == "__main__":
    main()

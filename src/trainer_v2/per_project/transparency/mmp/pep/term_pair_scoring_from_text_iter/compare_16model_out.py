import sys

from iter_util import load_jsonl
from misc_lib import SuccessCounter, two_digit_float
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep.term_pair_scoring_from_text_iter.read_segment_log import load_segment_log


def main():
    jsonl = load_jsonl(sys.argv[1])
    score_d = load_segment_log(jsonl)
    tsv_path = sys.argv[2]
    ret = list(tsv_iter(tsv_path))

    head = ["q_term", "d_term", "org_score", "new_score"]
    table = [head]
    suc = SuccessCounter()
    for q_term, d_term, score2 in ret:
        score2 = float(score2)
        score_list = score_d[q_term, d_term]
        score1 = score_list[0]
        if abs(score1 - score2) > 0.1:
            row = [q_term, d_term, two_digit_float(score1), two_digit_float(score2)]
            table.append(row)
            suc.fail()
        else:
            suc.suc()

    print_table(table)
    print("Success rate", suc.get_suc_prob())


if __name__ == "__main__":
    main()

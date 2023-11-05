import sys

from iter_util import load_jsonl
from misc_lib import SuccessCounter
from tab_print import tab_print
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep.term_pair_scoring_runner.read_segment_log import load_segment_log


def main():
    jsonl = load_jsonl(sys.argv[1])
    score_d = load_segment_log(jsonl)
    tsv_path = sys.argv[2]
    ret = list(tsv_iter(tsv_path))

    suc = SuccessCounter()
    for q_term, d_term, score2 in ret:
        score2 = float(score2)
        score_list = score_d[q_term, d_term]
        score1 = score_list[0]
        if abs(score1- score2) > 0.1:
            tab_print(q_term, d_term, score1, score2)
            suc.fail()
        else:
            suc.suc()

    print("Success rate", suc.get_suc_prob())


if __name__ == "__main__":
    main()

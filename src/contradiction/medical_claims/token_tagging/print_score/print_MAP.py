import os.path

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path, get_save_path2
from tab_print import print_table
from trainer_v2.chair_logging import c_log
from trec.trec_eval_wrap_fn import run_trec_eval_parse


class BioClaimMapCalc:
    def __init__(self, split):
        self.split = split
        self.qrel_path = get_sbl_qrel_path(split)
        self.expected_num_q_d = {
            ("val", "mismatch"): "209",
            ("val", "mismatch"): "209",
            ("test", "mismatch"): "311",
            ("test", "conflict"): "245",
        }

    def compute(self, run_name, tag):
        prediction_path = get_save_path2(run_name, tag)
        if os.path.exists(prediction_path):
            score_d = run_trec_eval_parse(prediction_path, self.qrel_path)
            s = score_d["map"]
            n_q = score_d["num_q"]
            n_q_expected = self.expected_num_q_d[self.split, tag]
            if n_q != n_q_expected:
                c_log.warning(f"Run {run_name} has {n_q} "
                              f"queries while {n_q_expected} is expected")
        else:
            s = "-"
        return s


def show_for_mismatch():
    run_list = ["random", "exact_match",
                "coattention", "lime", "word_seg", "word2vec_em",
                "psearch", "nlits86", "nlits87", "davinci"
                ]
    split = "test"
    scorer = BioClaimMapCalc(split)

    head = ["run name", "mismatch", "conflict"]
    table = [head]
    for run_name in run_list:
        row = [run_name]
        for tag in ["mismatch", "conflict"]:
            s = scorer.compute(run_name, tag)
            row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_for_mismatch()
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_scores, run_eval_with_bm25t
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import eval_dev_mrr
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def cand_2_1():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    dataset = "dev_sample1000"
    cut = 0.1
    mapping_val = 0.1
    table_name = f"cand2_1_cut{cut}"
    run_name = f"bm25_{table_name}"
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)
    run_eval_with_bm25t(dataset, mapping, run_name)


def main_1000_eval():
    dataset = "dev_sample1000"
    table_name = "cand2_1"
    run_name = f"bm25_{table_name}"
    score = eval_dev_mrr(dataset, run_name)
    print(f"mrr:\t{score}")


def main():
    cand_2_1()


if __name__ == "__main__":
    main()

from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_candidate, run_eval_with_bm25t, \
    load_binary_mapping_from_align_candidate
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def cand_2():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.candidate_pair_path
    dataset = "dev_sample1000"
    table_name = f"cand2_1_raw"
    run_name = f"bm25t_{table_name}"
    mapping = load_binary_mapping_from_align_candidate(table_path)
    run_eval_with_bm25t(dataset, mapping, run_name)


def main():
    cand_2()


if __name__ == "__main__":
    main()

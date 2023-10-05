from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from trainer_v2.per_project.transparency.mmp.bm25t_runner.common import bm25t_nltk_stem_rerank_run_and_eval, \
    bm25t_rerank_run_and_eval_from_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def main():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    table_name = f"cand2_1_corr"
    dataset = TREC_DL_2019
    bm25t_rerank_run_and_eval_from_scores(dataset, table_name, table_path)


if __name__ == "__main__":
    main()

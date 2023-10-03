from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from trainer_v2.per_project.transparency.mmp.bm25t_runner.common import bm25t_rerank_run_and_eval_from_list
from cpath import output_path
from misc_lib import path_join


def main():
    table_path = path_join(output_path, "msmarco", "passage", "align_candidates", "candidate2_1_pos.tsv")
    table_name = f"cand2_1_pos"
    dataset = TREC_DL_2019
    bm25t_rerank_run_and_eval_from_list(dataset, table_name, table_path)


if __name__ == "__main__":
    main()

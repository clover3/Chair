import sys

from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list, \
    load_second_column_from_bias_dir
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner
from typing import List
from cpath import output_path
from misc_lib import path_join


def load_car_bias_exp_resource():
    term_list = load_car_maker_list()
    term_list_set = set(term_list)


    query_list: List[str] = load_second_column_from_bias_dir("matched_query.tsv")
    passages = load_second_column_from_bias_dir("matched_doc.tsv")
    return passages, query_list, term_list_set, term_list


def main():
    model_path = sys.argv[1]
    job_no = int(sys.argv[2])

    n_per_job = 10
    score_log_f = open(path_join(output_path, "mmp", "bias", "score_log", f"{job_no}.tsv"), "w")
    st = n_per_job * job_no
    ed = st + n_per_job

    def doc_iter():
        for doc_idx in range(st, ed):
            doc_text = passages[doc_idx]
            yield doc_idx, doc_text

    passages, query_list, term_list_set, term_list = load_car_bias_exp_resource()
    run_inference_inner(doc_iter, model_path, query_list, term_list_set, term_list, score_log_f)



if __name__ == "__main__":
    main()



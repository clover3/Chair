import logging

from contradiction.medical_claims.retrieval.eval_helper import solve_bioclaim, batch_solve_bioclaim, \
    solve_bio_claim_and_save
from contradiction.medical_claims.retrieval.nli_system import get_nlits_relevance_module
from contradiction.medical_claims.retrieval.path_helper import get_retrieval_save_path
from trainer_v2.chair_logging import c_log
from trec.trec_parse import write_trec_ranked_list_entry

def exist_c_log():
    global c_lod


def nli_dev():
    module = get_nlits_relevance_module()
    split = "dev"
    run_name = "pep_devs"
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))




def main():
    c_log.setLevel(logging.DEBUG)
    nli_dev()


if __name__ == "__main__":
    main()
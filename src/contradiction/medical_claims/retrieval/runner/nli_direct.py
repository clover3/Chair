import logging

from contradiction.medical_claims.retrieval.eval_helper import solve_bioclaim, batch_solve_bioclaim
from contradiction.medical_claims.retrieval.nli_direct import NLIAsRelevance, get_entail_cont, get_entail
from contradiction.medical_claims.retrieval.path_helper import get_retrieval_save_path
from list_lib import right
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client, get_cached_client, \
    get_nli14_cache_client
from trec.trec_parse import write_trec_ranked_list_entry





def do_inner(run_name, split, score_getter):
    c_log.info(f"nli_drect({run_name}, {split})")
    nli_predict_fn = get_nli14_cache_client()
    module = NLIAsRelevance(nli_predict_fn, score_getter)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


def main():
    c_log.setLevel(logging.DEBUG)
    split = "test"

    run_name = "nli_direct_ec"
    score_getter = get_entail_cont
    do_inner(run_name, split, score_getter)

    run_name = "nli_direct_e"
    score_getter = get_entail
    do_inner(run_name, split, score_getter)


if __name__ == "__main__":
    main()

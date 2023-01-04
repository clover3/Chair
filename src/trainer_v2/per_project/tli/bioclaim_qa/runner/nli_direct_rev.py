import logging

from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import solve_bioclaim, batch_solve_bioclaim
from trainer_v2.per_project.tli.qa_scorer.nli_direct import NLIAsRelevance, get_entail_cont, get_entail, \
    NLIAsRelevanceRev
from trainer_v2.per_project.tli.bioclaim_qa.path_helper import get_retrieval_save_path
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_nli14_cache_client
from trec.trec_parse import write_trec_ranked_list_entry


def do_inner(run_name, split, score_getter):
    c_log.info(f"nli_drect({run_name}, {split})")
    nli_predict_fn = get_nli14_cache_client()
    module = NLIAsRelevanceRev(nli_predict_fn, score_getter)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    save_name = f"{run_name}_{split}"
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(save_name))


def main():
    c_log.setLevel(logging.DEBUG)
    run_name = "nli_direct_rev"
    score_getter = get_entail
    do_inner(run_name, "dev", score_getter)
    do_inner(run_name, "test", score_getter)


if __name__ == "__main__":
    main()

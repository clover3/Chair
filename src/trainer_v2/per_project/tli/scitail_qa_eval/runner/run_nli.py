import logging

from trainer_v2.per_project.tli.qa_scorer.nli_direct import NLIAsRelevance, get_entail_cont, get_entail, \
    NLIAsRelevanceRev
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_nli14_cache_client
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import batch_solve_save_scitail_qa


def do_inner(run_name, score_getter):
    c_log.info(f"nli_drect({run_name})")
    nli_predict_fn = get_nli14_cache_client()
    module = NLIAsRelevance(nli_predict_fn, score_getter)
    batch_solve_save_scitail_qa(module.batch_predict, run_name)


def main():
    c_log.setLevel(logging.DEBUG)
    run_name = "nli_direct_e"
    score_getter = get_entail
    do_inner(run_name, score_getter)


def do_inner_rev(run_name, score_getter):
    c_log.info(f"nli_drect_rev({run_name})")
    nli_predict_fn = get_nli14_cache_client()
    module = NLIAsRelevanceRev(nli_predict_fn, score_getter)
    batch_solve_save_scitail_qa(module.batch_predict, run_name)


def main():
    c_log.setLevel(logging.DEBUG)
    run_name = "nli_rev_direct"
    score_getter = get_entail
    do_inner_rev(run_name, score_getter)


if __name__ == "__main__":
    main()

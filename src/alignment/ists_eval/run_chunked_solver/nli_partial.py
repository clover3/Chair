import os

from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.nli_partial import PartialNLIDrivenSolver
from cpath import data_path
from taskman_client.wrapper3 import TaskReporting
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, \
    get_pep_cache_client, get_nli14_cache_client

word2vec_path = os.path.join(data_path, "GoogleNews-vectors-negative300.bin")


def run_solver(nli_predictor, nli_type):
    run_name = "{}_chunked".format(nli_type)
    solver = PartialNLIDrivenSolver(nli_predictor, word2vec_path)
    c_log.info("Running predictions for {}".format(run_name))
    chunked_solve_and_save_eval(solver, run_name, "headlines", "train")


def pep():
    nli_type = "pep_nli_partial"
    nli_predictor: NLIPredictorSig = get_pep_cache_client()
    run_solver(nli_predictor, nli_type)


def main_nli():
    nli_type = "base_nli_partial"
    with TaskReporting(nli_type):
        nli_predictor: NLIPredictorSig = get_nli14_cache_client()
        run_solver(nli_predictor, nli_type)


if __name__ == "__main__":
    main_nli()

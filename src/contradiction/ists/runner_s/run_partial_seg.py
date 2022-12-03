import os

from contradiction.ists.predict_common import eval_ists_noali_headlines_train_batch
from contradiction.medical_claims.token_tagging.batch_solver_common import BatchSolver
from contradiction.medical_claims.token_tagging.solvers.partial_search_solver import PartialSegSolvingAdapter2
from trainer_v2.keras_server.name_short_cuts import get_nli14_predictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def get_batch_partial_seg_solver(sel_score_fn) -> BatchSolver:
    predict_fn = get_nli14_predictor()
    adapter = PartialSegSolvingAdapter2(predict_fn, sel_score_fn)
    return BatchSolver(adapter)


def partial_seg(run_name):
    def sel_score_fn(probs):
        return probs[1]
    solver = get_batch_partial_seg_solver(sel_score_fn)
    eval_ists_noali_headlines_train_batch(run_name, solver)


def main2():
    run_name = "partial_seg"
    partial_seg(run_name)


if __name__ == "__main__":
    main2()

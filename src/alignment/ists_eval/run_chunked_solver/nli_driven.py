from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.exact_match_solver import ExactMatchSolver
from alignment.ists_eval.chunked_solver.nli_driven import NLIDrivenSolver
from trainer_v2.keras_server.name_short_cuts import get_nli14_predictor, NLIPredictorSig


def main():
    nli_type = "base_nli"
    nli_predictor: NLIPredictorSig = get_nli14_predictor()
    solver = NLIDrivenSolver(nli_predictor, ExactMatchSolver())
    chunked_solve_and_save_eval(solver, "{}_chunked".format(nli_type), "headlines", "train")


if __name__ == "__main__":
    main()
import os

from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.exact_match_solver import ExactMatchSolver
from alignment.ists_eval.chunked_solver.nli_driven import NLIDrivenSolver
from alignment.ists_eval.chunked_solver.word2vec_solver import Word2VecSolver
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_predictor, NLIPredictorSig, get_pep_client


def pep_word2vec():
    nli_type = "pep_word2vec"
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    nli_predictor: NLIPredictorSig = get_pep_client()
    solver = NLIDrivenSolver(nli_predictor, Word2VecSolver(word2vec_path))
    chunked_solve_and_save_eval(solver, "{}_chunked".format(nli_type), "headlines", "train")


def main_pep():
    nli_type = "pep"
    nli_predictor: NLIPredictorSig = get_pep_client()
    solver = NLIDrivenSolver(nli_predictor, ExactMatchSolver())
    chunked_solve_and_save_eval(solver, "{}_chunked".format(nli_type), "headlines", "train")


def main_nli():
    nli_type = "base_nli"
    nli_predictor: NLIPredictorSig = get_keras_nli_300_predictor()
    solver = NLIDrivenSolver(nli_predictor, ExactMatchSolver())
    chunked_solve_and_save_eval(solver, "{}_chunked".format(nli_type), "headlines", "train")


if __name__ == "__main__":
    pep_word2vec()

import os

from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.exact_match_solver import ExactMatchSolver
from alignment.ists_eval.chunked_solver.word2vec_solver import Word2VecSolver
from alignment.ists_eval.chunked_solver.coattention import get_co_attention_chunked_solver


def main():
    solver = ExactMatchSolver()
    chunked_solve_and_save_eval(solver, "exact_match_chunked", "headlines", "train")


def main():
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    solver = Word2VecSolver(word2vec_path)
    chunked_solve_and_save_eval(solver, "w2v_chunked", "headlines", "train")


if __name__ == "__main__":
    main()

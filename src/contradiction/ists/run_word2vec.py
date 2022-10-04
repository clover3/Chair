from contradiction.ists.predict_common import eval_ists_noali_headlines_train
from contradiction.medical_claims.token_tagging.solvers.word2vec_solver import get_word2vec_solver
from misc_lib import tprint


def main():
    run_name = "word2vec"
    tprint("Building solver")
    solver = get_word2vec_solver()
    tprint("Building solver DONE")
    eval_ists_noali_headlines_train(run_name, solver)


if __name__ == "__main__":
    main()

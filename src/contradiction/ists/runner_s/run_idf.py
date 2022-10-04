
from contradiction.ists.predict_common import eval_ists_noali_headlines_train
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import get_tf_idf_solver
from misc_lib import tprint


def main():
    run_name = "idf"
    tprint("Building solver")
    solver = get_tf_idf_solver()
    tprint("Building solver DONE")
    eval_ists_noali_headlines_train(run_name, solver)


if __name__ == "__main__":
    main()

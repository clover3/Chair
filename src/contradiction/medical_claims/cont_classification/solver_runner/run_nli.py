from contradiction.medical_claims.cont_classification.run_eval_fns import run_cont_solver_and_save, \
    run_cont_prob_solver_and_save
from contradiction.medical_claims.cont_classification.solvers.direct_nli import get_nli14_classifier, NLIWQuestion, \
    get_nli_q1, get_nli_q2, get_nli_q3, get_nli_q4
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_predictor


def main():
    for split in ["dev", "test"]:
        solver = get_nli14_classifier()
        run_name = "nli"
        run_cont_prob_solver_and_save(solver, run_name, split)


def main():
    for split in ["dev", "test"]:
        solver = get_nli_q1()
        run_name = "nli_q1"
        run_cont_prob_solver_and_save(solver, run_name, split)


def main():
    for split in ["dev", "test"]:
        solver = get_nli_q2()
        run_name = "nli_q2"
        run_cont_prob_solver_and_save(solver, run_name, split)


def main():
    for split in ["dev", "test"]:
        solver = get_nli_q3()
        run_name = "nli_q3"
        run_cont_prob_solver_and_save(solver, run_name, split)


def main():
    for split in ["dev", "test"]:
        solver = get_nli_q4()
        run_name = "nli_q4"
        run_cont_prob_solver_and_save(solver, run_name, split)


if __name__ == "__main__":
    main()
from contradiction.medical_claims.cont_classification.run_eval_fns import run_cont_solver_and_save
from contradiction.medical_claims.cont_classification.solvers.trivial_solver import Majority, RandomClassifier


def majority():
    split = "test"
    solver = Majority()
    run_name = "majority"
    run_cont_solver_and_save(solver, run_name, split)


def random_cls():
    split = "test"
    solver = RandomClassifier()
    run_name = "random"
    run_cont_solver_and_save(solver, run_name, split)


if __name__ == "__main__":
    solver_d = {
        'majority': Majority(),
        'random': RandomClassifier()
    }
    for split in ["dev", "test"]:
        for run_name, solver in solver_d.items():
            run_cont_solver_and_save(solver, run_name, split)

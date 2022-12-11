from contradiction.medical_claims.cont_classification.run_eval_fns import tune_scores, run_cont_solver_and_save
from contradiction.medical_claims.cont_classification.solvers.similarity_solver import BM25Classifier


def do_tune():
    split = "dev"
    solver = BM25Classifier(split)
    tune_scores(solver, split, "f1")


def dev_main():
    t = 30
    split = "dev"
    solver = BM25Classifier(split)
    solver.threshold = t
    run_name = "bm25_tuned"
    run_cont_solver_and_save(solver, run_name, split)



def test_main():
    t = 25
    split = "test"
    solver = BM25Classifier(split)
    solver.threshold = t
    run_name = "bm25_tuned"
    run_cont_solver_and_save(solver, run_name, split)


if __name__ == "__main__":
    dev_main()
    do_tune()
    test_main()
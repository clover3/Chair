from trainer_v2.per_project.tli.qa_scorer.tli_based import get_tli_based_models
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import batch_solve_save_scitail_qa


def main():
    run_name = "tnli2"
    batch_predict = get_tli_based_models("tli2")
    batch_solve_save_scitail_qa(batch_predict, run_name)


if __name__ == "__main__":
    main()
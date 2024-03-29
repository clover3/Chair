import sys
from typing import List

from alignment import MatrixScorerIF
from alignment.data_structure.eval_data_structure import Alignment2D
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.matrix_scorers.related_scoring_common import run_scoring
from alignment.nli_align_path_helper import load_mnli_rei_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer
from tlm.qtype.partial_relevance.runner.related_prediction.run_predict import get_method


def load_problem_run_scoring_and_save(dataset_name, method: str):
    split = "dev"
    print(f"load_problem_run_scoring_and_save(\"{dataset_name}\", \"{method}\")")
    scorer: MatrixScorerIF = get_method(method)
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(split)
    answers: List[Alignment2D] = run_scoring(problems, scorer)
    save_related_eval_answer(answers, dataset_name, method)


def main():
    method = sys.argv[1]
    load_problem_run_scoring_and_save("mnli_align", method)


if __name__ == "__main__":
    main()
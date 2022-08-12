from typing import List

from alignment.data_structure.eval_data_structure import Alignment2D
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.lexical_alignment.get_lexical_aligner import get_scorer
from alignment.matrix_scorers.related_scoring_common import run_scoring
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.related.related_answer_data_path_helper import save_related_eval_answer


def run_related_prediction_save(dataset_name, scorer_name):
    scorer: MatrixScorerIF = get_scorer(scorer_name)
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    answers: List[Alignment2D] = run_scoring(problems, scorer)
    save_related_eval_answer(answers, dataset_name, scorer_name)



def main():
    dataset = "train"
    # run_related_prediction_save(dataset, "random")
    # run_related_prediction_save(dataset, "segment_exact_match")
    # run_related_prediction_save(dataset, "all_zero")
    run_related_prediction_save(dataset, "lexical_v1")


if __name__ == "__main__":
    main()
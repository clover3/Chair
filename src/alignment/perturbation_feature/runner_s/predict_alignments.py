from typing import List

from alignment import RelatedEvalAnswer, RelatedEvalInstance
from alignment.matrix_scorers.related_scoring_common import run_scoring
from alignment.nli_align_path_helper import load_mnli_rei_problem, save_related_eval_answer
from alignment.perturbation_feature.learned_perturbation_scorer import LearnedPerturbationScorer, load_pert_pred_model


def main():
    dataset_name = "train_head"
    model_name = 'train_v1_1_2K_linear_9'
    scorer_name = "pert_" + model_name
    model = load_pert_pred_model(model_name, None)
    scorer = LearnedPerturbationScorer(model)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    problem_list = problem_list[:30]
    answers: List[RelatedEvalAnswer] = run_scoring(problem_list, scorer)
    save_related_eval_answer(answers, dataset_name, scorer_name)


if __name__ == "__main__":
    main()
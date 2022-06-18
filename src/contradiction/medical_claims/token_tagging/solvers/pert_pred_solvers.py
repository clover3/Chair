from typing import List, Tuple

from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.learned_perturbation_scorer import LearnedPerturbationScorer, load_pert_pred_model
from alignment.perturbation_feature.pert_model_1d import get_dataset, split_val
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d
from bert_api import SegmentedInstance
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
# Mismatch score: Highest align score for each tokens.
# Aligns scores:
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import get_tf_idf_solver
from contradiction.medical_claims.token_tagging.solvers.ensemble_solver import EnsembleSolver, NormalizeEnsembleScorer
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSTHandleSolver
from data_generator.tokenizer_wo_tf import get_tokenizer


class PertPredSolver(TokenScoringSolverIF):
    def __init__(self, scorer):
        self.tokenizer = get_tokenizer()
        self.scorer = scorer

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        return self._solve_inner(t1, t2), self._solve_inner(t2, t1)

    def _solve_inner(self, target: SegmentedText, other: SegmentedText):
        seg_inst = SegmentedInstance(target, other)
        contrib: ContributionSummary = self.scorer.eval_contribution(seg_inst)

        match_score = []
        for idx in target.enum_seg_idx():
            scores = contrib.table[idx]
            match_score.append(max(scores))
        mismatch_score = [1-s for s in match_score]
        return mismatch_score


def load_dataset():
    dataset_name = "train_v1_1_2K"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape_1d()
    dataset = get_dataset(shape, batch_size, input_file)
    train, val = split_val(dataset, int(2000 / 8))
    return val


def get_pert_pred_solver() -> TokenScoringSolverIF:
    print("pert_pred_solvers")
    model_name = 'train_v1_1_2K_linear_9'
    print("loading tf model")
    new_model = load_pert_pred_model(model_name, load_dataset())
    print("building solver")
    scorer = LearnedPerturbationScorer(new_model)
    pert = PertPredSolver(scorer)
    return pert


def get_pert_pred_solver_ex1():
    solver_list = [get_pert_pred_solver(), ExactMatchSTHandleSolver()]
    return EnsembleSolver(solver_list)


def get_pert_pred_solver_ex2():
    solver_list: List[TokenScoringSolverIF] = [ExactMatchSTHandleSolver(), get_pert_pred_solver()]
    return NormalizeEnsembleScorer(solver_list, [None, 0.1])


def get_pert_pred_solver_ex3():
    solver_list: List[TokenScoringSolverIF] = [get_tf_idf_solver(), get_pert_pred_solver()]
    return NormalizeEnsembleScorer(solver_list, [1, 1])



def main():
    get_pert_pred_solver()


if __name__ == "__main__":
    main()

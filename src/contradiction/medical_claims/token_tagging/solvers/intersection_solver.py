from typing import List, Tuple

from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText
from contradiction.medical_claims.token_tagging.intersection_search.deletion_tools import Subsequence
from contradiction.medical_claims.token_tagging.intersection_search.tree_deletion_search import TreeDeletionSearch
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from data_generator.tokenizer_wo_tf import get_tokenizer


def get_scores(n_tokens: int, subsequence: Subsequence) -> List[int]:
    assert n_tokens == subsequence.parent.get_seg_len()
    score_list = []
    for idx in range(n_tokens):
        if idx in subsequence.parent_drop_indices:
            score = 1
        else:
            score = 0
        score_list.append(score)
    return score_list


class IntersectionSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn, verbose):
        self.predict_fn = predict_fn
        self.deletion_search = TreeDeletionSearch(predict_fn, verbose)
        self.tokenizer = get_tokenizer()

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        t1_p: Subsequence = self.deletion_search.find_longest_non_neutral_subset(t2, t1)
        t2_p: Subsequence = self.deletion_search.find_longest_non_neutral_subset(t1, t2)
        return get_scores(len(text1_tokens), t1_p), get_scores(len(text2_tokens), t2_p)



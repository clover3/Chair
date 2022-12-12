import logging
from typing import List, Tuple

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from contradiction.medical_claims.token_tagging.intersection_search.dev_align import get_scores_by_many_perturbations, \
    enum_small_segments
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.solvers.align_to_mismatch import convert_align_to_mismatch
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer.promise import PromiseKeeper, MyPromise
from trainer_v2.chair_logging import c_log
import numpy as np

class SearchSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        c_log.setLevel(logging.WARN)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)

        scores2 = get_scores_by_many_perturbations(self.predict_fn, t1, t2)
        scores1 = get_scores_by_many_perturbations(self.predict_fn, t2, t1)
        return scores1, scores2


class PartialSegSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn, target_label):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        self.target_label = target_label
        c_log.setLevel(logging.WARN)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)

        scores2 = enum_small_segments(self.predict_fn, t1, t2, self.target_label)
        scores1 = enum_small_segments(self.predict_fn, t2, t1, self.target_label)
        return scores1, scores2


class WordSegAlignSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        c_log.setLevel(logging.WARN)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)

        pk = PromiseKeeper(self.predict_fn)
        f_mat = []
        for seg_idx1, seg_indices1 in enumerate(t1.seg_token_indices):
            s1 = t1.get_sliced_text(seg_indices1)
            f_array = []
            for seg_idx2, seg_indices2 in enumerate(t2.seg_token_indices):
                s2 = t2.get_sliced_text(seg_indices2)
                x = NLIInput(s1, s2)
                f = MyPromise(x, pk).future()
                f_array.append(f)
            f_mat.append(f_array)

        pk.do_duty()
        match_matrix = []
        for seg_idx1, seg_indices1 in enumerate(t1.seg_token_indices):
            f_array = f_mat[seg_idx1]

            def future_to_e_score(f):
                pred = f.get()
                entailment = pred[0]
                return entailment

            match_row = [future_to_e_score(f) for f in f_array]
            match_matrix.append(match_row)

        return convert_align_to_mismatch(np.array(match_matrix))


class WordSegSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        c_log.setLevel(logging.WARN)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        text1 = " ".join(text1_tokens)
        text2 = " ".join(text2_tokens)
        pairs: List[Tuple[str, str]] = []
        for token in text1_tokens:
            pairs.append((text2, token))
        for token in text2_tokens:
            pairs.append((text1, token))

        probs = self.predict_fn(pairs)

        i = 0
        scores1 = []
        for token in text1_tokens:
            pred = probs[i]
            scores1.append(pred[1])
            i += 1

        scores2 = []
        for token in text2_tokens:
            pred = probs[i]
            scores2.append(pred[1])
            i += 1

        return scores1, scores2

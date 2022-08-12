import abc
from typing import List, Dict, Tuple

import numpy as np

from alignment.data_structure.batch_scorer_if import BatchMatrixScorerIF, AlignProblem, AlignAnswer, BASAIF
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText, \
    merge_subtoken_level_scores
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_predictor import ClsProbePredictor, ProbeOutput
from list_lib import left, right, pairzip
from misc_lib import average


class ProbeAdapter(BASAIF):
    ChildItem = Tuple[NLIInput, int]

    def __init__(self, predictor, sel_score_fn):
        self.tokenizer = get_tokenizer()
        self.sel_score_fn = sel_score_fn
        self.predictor: ClsProbePredictor = predictor

    def neural_worker(self, items: List[ChildItem]) -> List[Tuple[ProbeOutput, int]]:
        def predict_one(nli_input: NLIInput) -> ProbeOutput:
            return self.predictor.predict(nli_input.prem.tokens_ids, nli_input.hypo.tokens_ids)

        input_input_list: List[NLIInput] = left(items)
        po_iter = map(predict_one, input_input_list)
        return pairzip(po_iter, right(items))

    def reduce(self, text1_tokens, text2_tokens, output: List[Tuple[ProbeOutput, int]]) -> List[List[float]]:
        # AlignScore[i,j] = \Avg_k Probe_neutral_k(p / p_i , h, j) - Probe_neutral_k(p, h, j)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)

        base_po, minus_one = output[0]

        layer_range = list(range(1, 13))

        assert minus_one == -1

        align_score_d: Dict[int, List[float]] = {}
        for po, i in output[1:]:
            maybe_t2_len = po.sep_idx2 - (po.sep_idx1 + 1)
            if len(t2.tokens_ids) != maybe_t2_len:
                print("WARNING length of {} is expected but got {}".format(len(t2.tokens_ids), maybe_t2_len))

            align_score_per_layer_list: List[np.array] = []
            for layer_no in layer_range:
                change = base_po.get_seg2_prob(layer_no) - po.get_seg2_prob(layer_no)
                align_score_per_layer: List[float] = [self.sel_score_fn(item_probe) for item_probe in change]
                align_score_per_layer_np = np.array(align_score_per_layer)
                align_score_per_layer_list.append(align_score_per_layer_np)

            align_score_per_layer_np = np.stack(align_score_per_layer_list, 0)
            align_score: np.array = np.average(align_score_per_layer_np, 0)
            align_score_l: List[float] = merge_subtoken_level_scores(average, align_score.tolist(), t2)
            align_score_d[i] = align_score_l

        return [align_score_d[i] for i in range(len(text1_tokens))]

    def enum_child(self, text1_tokens: List[str], text2_tokens: List[str]) -> List[ChildItem]:
        # Generate instances with t1 perturbed.
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        es_list = []  # es = Encoded Segment
        es_list.append((NLIInput(t1, t2), -1))
        for i in range(len(text1_tokens)):
            t1_sub = t1.get_dropped_text([i])
            es = (NLIInput(t1_sub, t2), i)
            es_list.append(es)
        return es_list


class BatchProbeSolver(BatchMatrixScorerIF):
    @abc.abstractmethod
    def solve(self, problem_list: List[AlignProblem]) -> List[AlignAnswer]:
        pass


class ProbeSolver(MatrixScorerIF2):
    def __init__(self, predictor):
        self.predictor: ClsProbePredictor = predictor

        def get_neutral(l):
            return l[0]

        self.probe_adapter = ProbeAdapter(self.predictor, get_neutral)
        self.tokenizer = get_tokenizer()

    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        payloads = self.probe_adapter.enum_child(tokens1, tokens2)
        n_output = self.probe_adapter.neural_worker(payloads)
        mat_scores = self.probe_adapter.reduce(tokens1, tokens2, n_output)
        return mat_scores


def get_probe_solver() -> MatrixScorerIF2:
    return ProbeSolver(ClsProbePredictor())

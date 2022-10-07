from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from contradiction.medical_claims.token_tagging.batch_solver_common import BSAdapterIF, BatchSolver, NeuralOutput
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import average
from trainer_v2.custom_loop.per_task.nli_ts_util import EncodedSegmentIF
from trainer_v2.custom_loop.per_task.nli_ms_util import LocalDecisionNLIMS, get_local_decision_nlims
from trainer_v2.custom_loop.run_config2 import RunConfig2


class TokenizedTextBasedES(EncodedSegmentIF):
    def __init__(self, tt1, tt2, x):
        self.tt1: TokenizedText = tt1
        self.tt2: TokenizedText = tt2
        self.x: List = x

    def get_input(self):
        return self.x


# NLIMS is a model which feed multiple segments of a text at a time.
# In NLITS80 model, each subword is directly feed. We will merge scores afterward
class NLISingleToken2DAdapter(BSAdapterIF):
    def __init__(self, nlims: LocalDecisionNLIMS, target_label: int):
        self.nlims: LocalDecisionNLIMS = nlims
        self.tokenizer = get_tokenizer()
        self.target_label = target_label

    def enum_child(self, t1: List[str], t2: List[str]):
        tt1 = TokenizedText.from_word_tokens(" ".join(t1), self.tokenizer, t1)
        tt2 = TokenizedText.from_word_tokens(" ".join(t2), self.tokenizer, t2)
        if len(tt2.sbword_mapping) > 100:
            print("Sequence has {} subwords, maybe this would be truncated".format(len(tt2.sbword_mapping)))
        x = self.nlims.encode_fn(tt1.sbword_tokens, tt2.sbword_tokens)
        es_list = [TokenizedTextBasedES(tt1, tt2, x)]
        return es_list

    def neural_worker(self, items: List[TokenizedTextBasedES]) -> List[Tuple[NeuralOutput, TokenizedTextBasedES]]:
        l_decisions = self.nlims.predict_es(items)
        return list(zip(l_decisions, items))

    def reduce(self, t1: List[str], t2: List[str],
               item: List[Tuple[NeuralOutput, TokenizedTextBasedES]]) -> List[float]:
        assert len(item) == 1
        neural_output, tt_based_es = item[0]
        local_decision = neural_output
        seq_len1, seq_len2, c = local_decision.shape
        # Originally g_decision is made by first reduce_max in p_side and them fuzzy sum over h_side
        word_idx_grouped: Dict[int, List[float]] = defaultdict(list)
        l2 = min(len(tt_based_es.tt2.sbword_mapping) + 2, seq_len2)
        for i in range(l2):
            sb_idx = i - 1
            decision_row = local_decision[:, i]
            three_scores = np.max(decision_row, axis=0)
            score_for_this_h_token: float = three_scores[self.target_label]
            try:
                word_idx = tt_based_es.tt2.sbword_mapping[sb_idx]
                word_idx_grouped[word_idx].append(score_for_this_h_token)
            except IndexError:
                pass

        merge_fn = average
        score_array = []
        for idx, _ in enumerate(t2):
            s = merge_fn(word_idx_grouped[idx])
            score_array.append(s)

        return score_array


def get_batch_solver_nli_single_token_2d(run_config: RunConfig2, target_label: int):
    nlims = get_local_decision_nlims(run_config)
    adapter = NLISingleToken2DAdapter(nlims, target_label)
    solver = BatchSolver(adapter)
    return solver


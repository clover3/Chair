from typing import List, Tuple

from contradiction.medical_claims.token_tagging.batch_solver_common import NeuralOutput, AdapterIF, BatchSolver
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import Averager
from trainer_v2.custom_loop.demo.demo_common import enum_hypo_token_tuple_from_tokens, EncodedSegmentIF
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore


class ESTwoPiece(EncodedSegmentIF):
    def __init__(self, input_x,
                 t1,
                 t2,
                 h_tokens_list,
                 st, ed):
        self.input_x = input_x
        self.t1 = t1
        self.t2 = t2
        self.h_tokens_list = h_tokens_list
        self.st = st
        self.ed = ed

    def get_input(self):
        return self.input_x


class ScoreReducer:
    def __init__(self, target_label: int):
        self.target_label = target_label

    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [Averager() for _ in text2_tokens]
        for probs, es_item in records:
            s = probs[self.target_label]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append(s)
        scores = [s.get_average() for s in scores_building]
        return scores


def get_batch_solver_nlits(run_name: str, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_name, encoder_name)
    adapter = NLITSAdapter(nlits, ScoreReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


class NLITSAdapter(AdapterIF):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 score_reducer: ScoreReducer):
        self.nlits: LocalDecisionNLICore = nlits
        self.tokenizer = get_tokenizer()
        self.score_reducer = score_reducer

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        es_list = []
        for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer, t2, 3):
            x = self.nlits.encode_fn(p_tokens, h_first, h_second)
            es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
            es_list.append(es)
        return es_list

    def neural_worker(self, items: List[ESTwoPiece]) -> List[Tuple[NeuralOutput, ESTwoPiece]]:
        l_decisions = self.nlits.predict_es(items)
        return list(zip(l_decisions, items))

    def reduce(self, t1, t2, item: List[Tuple[NeuralOutput, ESTwoPiece]]) -> List[float]:
        return self.score_reducer.reduce(item, t2)


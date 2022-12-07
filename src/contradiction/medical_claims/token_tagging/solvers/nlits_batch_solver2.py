from contradiction.medical_claims.token_tagging.batch_solver_common import BSAdapterIF, NeuralOutput, BatchSolver
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import NLITSAdapter, ScoreReducerI, \
    ESTwoPiece
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore, enum_hypo_token_tuple_from_tokens
from typing import List, Iterable, Callable, Dict, Tuple, Set, TypeVar

from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.cip.cip_module import get_cip3

CIPInput = TypeVar("CIPInput")

def es_to_cip(tokenizer, e: ESTwoPiece):
    h_tokens = list(flatten(map(tokenizer.tokenize, e.t2)))
    first, second = e.h_tokens_list
    return h_tokens, first, second


class NLITSAdapterCIP(BSAdapterIF):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 cip: Callable,
                 score_reducer: ScoreReducerI):
        self.nlits: LocalDecisionNLICore = nlits
        self.tokenizer = get_tokenizer()
        self.score_reducer = score_reducer
        self.es_to_cip: Callable[[ESTwoPiece], CIPInput] = lambda e: es_to_cip(self.tokenizer, e)
        self.cip: Callable[[Iterable[CIPInput]], List[float]] = cip

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        n_seg = len(t2)
        es_list = []
        for offset in [0, 1, 2]:
            for window_size in [1, 3, 6]:
                if window_size >= n_seg:
                    break
                for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(
                        self.tokenizer,
                        t2, window_size, offset):
                    x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                    es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
                    es_list.append(es)
                    x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                    es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
                    es_list.append(es)
        return es_list

    def neural_worker(self, items: List[ESTwoPiece]) -> List[Tuple[NeuralOutput, float, ESTwoPiece]]:
        cip_inputs = map(self.es_to_cip, items)
        cip_outputs = self.cip(cip_inputs)
        l_decisions = self.nlits.predict_es(items)
        return list(zip(l_decisions, cip_outputs, items))

    def reduce(self, t1, t2, item: List[Tuple[NeuralOutput, float, ESTwoPiece]]) -> List[float]:
        return self.score_reducer.reduce(item, t2)


class WeightedReducer(ScoreReducerI):
    def __init__(self, target_label: int, reduce_fn):
        self.target_label = target_label
        self.reduce_fn = reduce_fn

    def reduce(self, records: List[Tuple[List[float], float, ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [list() for _ in text2_tokens]
        for probs, weight, es_item in records:
            s = probs[self.target_label]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append((s, weight))

        scores = [self.reduce_fn(scores) for scores in scores_building]
        return scores


def weighted_sum_neg(items: List[Tuple[float, float]]):
    ws = 0
    denom = 0
    for value, raw_weight in items:
        weight = 1 - raw_weight
        ws += value * weight
        denom += weight
    return ws / denom


def get_batch_solver_nlits_cip(run_config: RunConfig2, encoder_name: str, target_label):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    cip = get_cip3(run_config)
    reducer = WeightedReducer(target_label, weighted_sum_neg)
    adapter = NLITSAdapterCIP(nlits, cip, reducer)
    solver = BatchSolver(adapter)
    return solver

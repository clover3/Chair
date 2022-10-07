from abc import ABC, abstractmethod
from typing import List, Tuple

from contradiction.medical_claims.token_tagging.batch_solver_common import NeuralOutput, BSAdapterIF, BatchSolver, \
    ECCOutput, ECCInput, BatchTokenScoringSolverIF
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import average
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore, enum_hypo_token_tuple_from_tokens, \
    EncodedSegmentIF
from trainer_v2.custom_loop.run_config2 import RunConfig2


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


class ScoreReducerI(ABC):
    @abstractmethod
    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        pass


class ReducerCommon(ScoreReducerI):
    def __init__(self, target_label: int, reduce_fn):
        self.target_label = target_label
        self.reduce_fn = reduce_fn

    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [list() for _ in text2_tokens]
        for probs, es_item in records:
            s = probs[self.target_label]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append(s)

        scores = [self.reduce_fn(scores) for scores in scores_building]
        return scores


class ReducerEC(ScoreReducerI):
    def __init__(self, reduce_fn):
        self.reduce_fn = reduce_fn

    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [list() for _ in text2_tokens]
        for probs, es_item in records:
            s = probs[1] + probs[2]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append(s)

        scores = [self.reduce_fn(scores) for scores in scores_building]
        return scores


class MinReducer(ReducerCommon):
    def __init__(self, target_label: int):
        self.target_label = target_label
        super(MinReducer, self).__init__(target_label, min)


class AvgReducer(ReducerCommon):
    def __init__(self, target_label: int):
        self.target_label = target_label
        super(AvgReducer, self).__init__(target_label, average)


class AlphaMinReducer(MinReducer):
    def __init__(self, target_label: int):
        self.target_label = target_label
        super(MinReducer, self).__init__(target_label, self.reduce_fn)

    def reduce_fn(self, items: List[float]):
        if not items:
            return 0
        items.sort()
        factor = 1
        accum = 0
        denom = 0
        for i in range(4):
            if i >= len(items):
                break
            accum += items[i] * factor
            denom += factor
            factor = factor / 2
        return accum / denom


def get_batch_solver_nlits(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter(nlits, AvgReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits2(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter2(nlits, AvgReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits3(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter2(nlits, MinReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits4(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter2(nlits, AlphaMinReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits5(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter3(nlits, MinReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits6(run_config: RunConfig2, encoder_name: str, target_label: int):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter3(nlits, AvgReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def get_batch_solver_nlits7(run_config: RunConfig2, encoder_name: str):
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    adapter = NLITSAdapter3(nlits, ReducerEC(average))
    solver = BatchSolver(adapter)
    return solver


class NLITSAdapter(BSAdapterIF):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 score_reducer: ScoreReducerI):
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


class NLITSAdapter2(NLITSAdapter):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 score_reducer: ScoreReducerI):
        super(NLITSAdapter2, self).__init__(nlits, score_reducer)

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        n_seg = len(t2)
        es_list = []
        for offset in [0, 1, 2]:
            for window_size in [3, 7, 11]:
                if window_size >= n_seg:
                    break
                for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer,
                                                                                   t2, window_size, offset):
                    x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                    es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
                    es_list.append(es)
        return es_list


class NLITSAdapter3(NLITSAdapter):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 score_reducer: ScoreReducerI):
        super(NLITSAdapter3, self).__init__(nlits, score_reducer)

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        n_seg = len(t2)
        es_list = []
        for offset in [0, 1, 2]:
            for window_size in [1, 3, 6]:
                if window_size >= n_seg:
                    break
                for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer,
                                                                                   t2, window_size, offset):
                    x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                    es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
                    es_list.append(es)
        return es_list


class SolverPostProcessorPunct(BatchTokenScoringSolverIF):
    def __init__(self, base_batch_solver: BatchSolver):
        self.base_solver = base_batch_solver

    def solve(self, payload: List[ECCInput]) -> List[ECCOutput]:
        scores_list = self.base_solver.solve(payload)
        out_scores_list = []
        for scores, problem in zip(scores_list, payload):
            scores1, scores2 = scores
            tokens1, tokens2 = problem

            for idx, t in enumerate(tokens1):
                if t in ",.;L'`!?":
                    if t not in tokens2:
                        scores1[idx] = 0.9

            for idx, t in enumerate(tokens2):
                if t in ",.;L'`!?":
                    if t not in tokens1:
                        scores2[idx] = 0.9

            out_scores_list.append((scores1, scores2))
        return out_scores_list

from contradiction.medical_claims.token_tagging.batch_solver_common import BSAdapterIF, NeuralOutput
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import NLITSAdapter, ScoreReducerI, \
    ESTwoPiece
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore, enum_hypo_token_tuple_from_tokens
from typing import List, Iterable, Callable, Dict, Tuple, Set



class NLITSAdapterCIP(BSAdapterIF):
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

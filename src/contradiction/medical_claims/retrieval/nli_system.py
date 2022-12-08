from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import average
from port_info import LOCAL_DECISION_PORT
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client


class NLIBasedRelevance:
    def __init__(self, nli_predict_fn: NLIPredictorSig):
        self.nli_predict_fn = nli_predict_fn

    def nli_calc(self, pair: List[Tuple[str, str]]) -> List[List[float]]:
        return self.nli_predict_fn(pair)

    def predict(self, query: str, document: str) -> float:
        q_tokens = query.split()
        pk = PromiseKeeper(self.nli_predict_fn)

        f_list = []
        for t in q_tokens:
            f: MyFuture[List[float]] = pk.get_future((document, t))
            f_list.append(f)

        pk.do_duty()
        nli_preds: List[List[float]] = list_future(f_list)

        def get_entail(probs: List[float]):
            return probs[0]

        entail_scores = list(map(get_entail, nli_preds))

        msg_tokens = ["{0}({1:.2f}".format(t, s) for t, s in zip(q_tokens, entail_scores)]
        msg = " ".join(msg_tokens)
        c_log.debug(msg)
        return average(entail_scores)

    def batch_predict(self, q_d_pairs: List[Tuple[str, str]]) -> List[float]:
        pk = PromiseKeeper(self.nli_predict_fn)
        NLIPred = List[float]

        q_docs_future = []
        for q, d in q_d_pairs:
            q_tokens = q.split()
            f_list: List[MyFuture[NLIPred]] = []
            for t in q_tokens:
                f: MyFuture[NLIPred] = pk.get_future((d, t))
                f_list.append(f)
            q_docs_future.append((q, f_list))

        def get_entail(probs: NLIPred):
            return probs[0]

        pk.do_duty()
        output = []
        for q, f_list in q_docs_future:
            nli_preds: List[NLIPred] = list_future(f_list)
            entail_scores = list(map(get_entail, nli_preds))
            s: float = average(entail_scores)
            output.append(s)
        return output


def get_nlits_relevance_module() -> NLIBasedRelevance:
    return NLIBasedRelevance(get_pep_client())
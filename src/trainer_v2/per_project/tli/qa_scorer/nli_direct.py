from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig

NLIPred = List[float]


class NLIAsRelevance:
    def __init__(
            self,
            nli_predict_fn: NLIPredictorSig,
            probs_to_score
    ):
        self.nli_predict_fn = nli_predict_fn
        self.probs_to_score = probs_to_score

    def nli_calc(self, pair: List[Tuple[str, str]]) -> List[List[float]]:
        return self.nli_predict_fn(pair)

    def batch_predict(self, q_d_pairs: List[Tuple[str, str]]) -> List[float]:
        pk = PromiseKeeper(self.nli_predict_fn)

        q_docs_future = []
        for q, d in q_d_pairs:
            f = pk.get_future((q, d))
            q_docs_future.append((q, f))

        pk.do_duty()
        output = []
        for q, f in q_docs_future:
            probs: NLIPred = f.get()

            output.append(self.probs_to_score(probs))
        return output


class NLIAsRelevanceRev:
    def __init__(
            self,
            nli_predict_fn: NLIPredictorSig,
            probs_to_score
    ):
        self.inner = NLIAsRelevance(nli_predict_fn, probs_to_score)

    def batch_predict(self, q_d_pairs: List[Tuple[str, str]]) -> List[float]:
        d_q_pairs = [(d, q) for q, d in q_d_pairs]
        return self.inner.batch_predict(d_q_pairs)


def get_entail_cont(probs: NLIPred):
    return probs[0] + probs[1]


def get_entail(probs: NLIPred):
    return probs[0]


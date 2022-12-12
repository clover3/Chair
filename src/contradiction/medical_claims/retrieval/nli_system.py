from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from adhoc.bm25_class import BM25
from misc_lib import average, weighted_sum
from port_info import LOCAL_DECISION_PORT
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client


class NLIBasedRelevance:
    def __init__(
            self,
            nli_predict_fn: NLIPredictorSig,
            bm25: BM25 = None
    ):
        self.nli_predict_fn = nli_predict_fn
        self.bm25 = bm25

    def nli_calc(self, pair: List[Tuple[str, str]]) -> List[List[float]]:
        return self.nli_predict_fn(pair)

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
            q_tokens = q.split()
            assert len(entail_scores) == len(q_tokens)

            weight_list = list(map(self.get_idf_for_token, q_tokens))
            s: float = weighted_sum(entail_scores, weight_list)
            output.append(s)
        return output

    def get_idf_for_token(self, token):
        if self.bm25 is None:
            return 1

        bm25_tokens = self.bm25.tokenizer.tokenize_stem(token)
        idf_sum = 0
        for t in bm25_tokens:
            idf_sum += self.bm25.term_idf_factor(t)
        return idf_sum


def enum_subseq(tokens_length: int, window_size, offset=0) -> Iterator[Tuple[int, int]]:
    st = offset
    while st < tokens_length:
        ed = min(st + window_size, tokens_length)
        yield st, ed
        st += window_size


def enum_subseq_136(tokens_length: int) -> Iterator[Tuple[int, int]]:
    for offset in [0, 1, 2]:
        for window_size in [1, 3, 6]:
            yield from enum_subseq(tokens_length, window_size, offset)


def token_level_attribution(scores: List[float], intervals: List[Tuple[int, int]]) -> List[float]:
    scores_building = defaultdict(list)

    for s, (st, ed) in zip(scores, intervals):
        for i in range(st, ed):
            scores_building[i].append(s)

    n_seq = max(scores_building.keys()) + 1
    scores = []
    for i in range(n_seq):
        scores.append(average(scores_building[i]))
    return scores




class NLIBasedRelevanceMultiSeg:
    def __init__(
            self,
            nli_predict_fn: NLIPredictorSig,
            bm25: BM25 = None
    ):
        self.nli_predict_fn = nli_predict_fn
        self.bm25 = bm25

    def nli_calc(self, pair: List[Tuple[str, str]]) -> List[List[float]]:
        return self.nli_predict_fn(pair)

    def batch_predict(self, q_d_pairs: List[Tuple[str, str]]) -> List[float]:
        pk = PromiseKeeper(self.nli_predict_fn)
        NLIPred = List[float]

        n_item = 0
        q_docs_future = []
        for q, d in q_d_pairs:
            q_tokens = q.split()
            f_list: List[MyFuture[NLIPred]] = []
            for st, ed in enum_subseq_136(len(q_tokens)):
                t = " ".join(q_tokens[st:ed])
                f: MyFuture[NLIPred] = pk.get_future((d, t))
                f_list.append(f)
                n_item += 1
            q_docs_future.append((q, f_list))

        def get_entail(probs: NLIPred):
            return probs[0]
        print("{} items".format(n_item))
        pk.do_duty()
        output = []
        for q, f_list in q_docs_future:
            nli_preds: List[NLIPred] = list_future(f_list)
            entail_scores = list(map(get_entail, nli_preds))
            q_tokens = q.split()

            subseq_list: List[Tuple[int, int]] = list(enum_subseq_136(len(q_tokens)))
            assert len(entail_scores) == len(subseq_list)

            token_level_scores: List[float] = token_level_attribution(entail_scores, subseq_list)
            weight_list = list(map(self.get_idf_for_token, q_tokens))
            s: float = weighted_sum(token_level_scores, weight_list)
            output.append(s)
        return output

    def get_idf_for_token(self, token):
        if self.bm25 is None:
            return 1

        bm25_tokens = self.bm25.tokenizer.tokenize_stem(token)
        idf_sum = 0
        for t in bm25_tokens:
            idf_sum += self.bm25.term_idf_factor(t)
        return idf_sum


def get_nlits_relevance_module() -> NLIBasedRelevance:
    return NLIBasedRelevance(get_pep_client())
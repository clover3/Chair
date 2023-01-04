from typing import List, Tuple

from adhoc.bm25_class import BM25
from misc_lib import weighted_sum
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136, token_level_attribution


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
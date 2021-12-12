from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput, SWTTTokenScorerInput
from misc_lib import TEL
from trainer.promise import MyFuture, MyPromise, PromiseKeeper
from trec.types import DocID, TrecRankedListEntry


class FutureScorerI(ABC):
    @abstractmethod
    def get_score_future(self, query_text: str,
                         doc: SegmentwiseTokenizedText,
                         passages: List[PassageRange]) -> MyFuture[SWTTScorerOutput]:
        pass

    @abstractmethod
    def do_duty(self):
        pass


class FutureScorerTokenBased(FutureScorerI):
    def __init__(self,
                 predictor,
                 ):
        self.predictor = predictor
        self.pk = PromiseKeeper(self._score_inner)

    def _score_inner(self, l: List[SWTTTokenScorerInput]) -> List[SWTTScorerOutput]:
        output_list: List[SWTTScorerOutput] = []
        for e in TEL(l):
            payload_list: List[Tuple[str, List[List[str]]]] = e.payload_list
            scores: List[float] = self.predictor.predict(payload_list)
            windows_st_ed_list: List[Tuple[SWTTIndex, SWTTIndex]] = e.windows_st_ed_list
            output: SWTTScorerOutput = SWTTScorerOutput(windows_st_ed_list,
                                                        scores,
                                                        e.doc)
            output_list.append(output)
        return output_list

    def get_score_future(self, query_text: str,
                         doc: SegmentwiseTokenizedText,
                         passages: List[PassageRange]) -> MyFuture[SWTTScorerOutput]:
        payload_list: List[Tuple[str, List[List[str]]]] = []
        for window_idx, window in enumerate(passages):
            st, ed = window
            e: Tuple[str, List[List[str]]] = (query_text, doc.get_word_tokens_grouped(st, ed))
            payload_list.append(e)
        promise_input: SWTTTokenScorerInput = SWTTTokenScorerInput(passages, payload_list, doc)
        promise = MyPromise(promise_input, self.pk)
        return promise.future()

    def do_duty(self):
        self.pk.do_duty()


def rerank_passages(doc_passage_d: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]],
                    ranked_list_groups,
                    query_list: List[Tuple[str, str]],
                    scorer: FutureScorerI,
                    ) \
        -> List[Tuple[str, List[Tuple[str, MyFuture]]]]:
    output: List[Tuple[str, List[Tuple[str, MyFuture]]]] = []
    for qid, query_text in query_list:
        ranked_list = ranked_list_groups[qid]
        doc_id_list = list(map(TrecRankedListEntry.get_doc_id, ranked_list))

        def do_score(doc_id) -> Tuple[str, MyFuture[SWTTScorerOutput]]:
            doc, passages = doc_passage_d[doc_id]
            score_future: MyFuture[SWTTScorerOutput] = scorer.get_score_future(query_text, doc, passages)
            return doc_id, score_future

        res: List[Tuple[str, MyFuture]] = list(map(do_score, doc_id_list))
        output.append((qid, res))
    scorer.do_duty()
    return output


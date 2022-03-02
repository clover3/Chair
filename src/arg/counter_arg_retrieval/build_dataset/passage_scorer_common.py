import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput, SWTTTokenScorerInput
from list_lib import lmap
from misc_lib import TEL
from trainer.promise import MyFuture, PromiseKeeper, MyPromise
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


def scoring_output_to_json(output: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]):
    def doc_and_score_to_json(doc_n_score: Tuple[str, SWTTScorerOutput]):
        doc_id, score = doc_n_score
        return {
            'doc_id': doc_id,
            'score': score.to_json()
        }

    def qid_and_docs_to_json(qid_and_doc: Tuple[str, List[Tuple[str, SWTTScorerOutput]]]):
        qid, docs = qid_and_doc
        docs_json = list(map(doc_and_score_to_json, docs))
        return {
            'qid': qid,
            'docs': docs_json
        }
    return lmap(qid_and_docs_to_json, output)


def json_to_scoring_output(j) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
    def parse_docs_and_scores(j_docs_and_scores) -> Tuple[str, SWTTScorerOutput]:
        return j_docs_and_scores['doc_id'], SWTTScorerOutput.from_json(j_docs_and_scores['score'])

    def parse_qid_and_docs(j_qid_and_docs) -> Tuple[str, List[Tuple[str, SWTTScorerOutput]]]:
        return j_qid_and_docs['qid'], lmap(parse_docs_and_scores, j_qid_and_docs['docs'])

    return lmap(parse_qid_and_docs, j)


def load_json_file_convert_to_scoring_output(path) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
    j = json.load(open(path, "r"))
    return json_to_scoring_output(j)


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
            try:
                doc, passages = doc_passage_d[doc_id]
            except KeyError as e:
                print("warning document {} is not found".format(doc_id))
                doc = SegmentwiseTokenizedText([])
                passages = []

            score_future: MyFuture[SWTTScorerOutput] = scorer.get_score_future(query_text, doc, passages)
            return doc_id, score_future

        res: List[Tuple[str, MyFuture]] = list(map(do_score, doc_id_list))
        output.append((qid, res))
    scorer.do_duty()
    return output


def unpack_futures(future_output) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
    output: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = []
    for ca_query, docs_and_scores_future in future_output:
        docs_and_scores = [(doc_id, doc_scores_future.get())
                           for doc_id, doc_scores_future in docs_and_scores_future]
        output.append((ca_query, docs_and_scores))
    return output


class PassageScoringInner:
    def __init__(self, scorer: FutureScorerI, rlg, doc_as_passage_dict):
        self.rlg = rlg
        self.doc_as_passage_dict = doc_as_passage_dict
        self.scorer: FutureScorerI = scorer

    def work(self, query_list) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
        future_output = rerank_passages(self.doc_as_passage_dict, self.rlg, query_list, self.scorer)
        output = unpack_futures(future_output)
        return output

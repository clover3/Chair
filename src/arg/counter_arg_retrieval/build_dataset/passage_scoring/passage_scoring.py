from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI, rerank_passages
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from list_lib import lmap


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


class PassageScoringInner:
    def __init__(self, scorer: FutureScorerI, rlg, doc_as_passage_dict):
        self.rlg = rlg
        self.doc_as_passage_dict = doc_as_passage_dict
        self.scorer: FutureScorerI = scorer

    def work(self, query_list) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
        future_output = rerank_passages(self.doc_as_passage_dict, self.rlg, query_list, self.scorer)
        output: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = []
        for ca_query, docs_and_scores_future in future_output:
            docs_and_scores = [(doc_id, doc_scores_future.get())
                               for doc_id, doc_scores_future in docs_and_scores_future]
            output.append((ca_query, docs_and_scores))
        return output
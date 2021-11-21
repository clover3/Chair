import os
from typing import List, Tuple, Iterator

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from bert_api.doc_score_defs import DocumentScorerOutput
from bert_api.doc_score_helper import RemoteDocumentScorer
from bert_api.msmarco_rerank import get_msmarco_client
from cache import load_from_pickle
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from misc_lib import TimeEstimator
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def run_relevance_classifier(ca_query_list: List[Tuple[str, str]])\
        -> Iterator[Tuple[str, List[Tuple[str, DocumentScorerOutput]]]]:
    client = get_msmarco_client()
    docs: List[Tuple[str, TokenizedText]] = load_from_pickle("ca_run3_document_processed")
    docs_d = dict(docs)
    document_scorer = RemoteDocumentScorer(client, 20)
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.txt")
    ranked_list_groups = load_ranked_list_grouped(rlg_path)

    # output: List[Tuple[str, List[Tuple[str, DocumentScorerOutput]]]] = []
    ticker = TimeEstimator(len(ranked_list_groups))
    for qid, query_text in ca_query_list:
        ranked_list = ranked_list_groups[qid]
        doc_id_list = list(map(TrecRankedListEntry.get_doc_id, ranked_list))

        def do_score(doc_id) -> Tuple[str, DocumentScorerOutput]:
            doc = docs_d[doc_id]
            sdp = document_scorer.score_relevance(query_text, doc)
            document_scorer.pk.do_duty()
            dso: DocumentScorerOutput = DocumentScorerOutput.from_dsos(sdp.get(), doc)
            return doc_id, dso

        res: Iterator[Tuple[str, DocumentScorerOutput]] = map(do_score, doc_id_list)
        yield qid, res
        ticker.tick()

    # return output


def main():
    # tsv_path = os.path.join(output_path, "ca_building", "run3", "crowd_ca_queries.txt")
    tsv_path = os.path.join(output_path, "ca_building", "run3", "my_ca_queries.txt")
    ca_query_list = load_ca_query_from_tsv(tsv_path)
    ca_d = {cq.qid: cq for cq in ca_query_list}
    ca_qid_query_list = [(cq.qid, cq.ca_query) for cq in ca_query_list]
    output = run_relevance_classifier(ca_qid_query_list)

    n_q_skip = 0
    skip_idx = 0
    for qid, docs_and_scores in output:
        ca_query: CAQuery = ca_d[qid]
        print(ca_query)
        if skip_idx < n_q_skip:
            print("skip")
            skip_idx += 1
            continue

        for doc_id, doc_scores in docs_and_scores:
            high_scores = list(filter(lambda x: x > 0.5, doc_scores.scores))
            print(f"{doc_id}: {len(high_scores)} / {len(doc_scores)}")


if __name__ == "__main__":
    main()

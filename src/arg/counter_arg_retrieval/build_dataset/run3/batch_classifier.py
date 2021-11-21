import os
import pickle
import sys
from typing import List, Tuple, Iterator

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from bert_api.doc_score_defs import DocumentScorerOutput
from bert_api.msmarco_local import DocumentScorer
from bert_api.predictor import Predictor
from cache import load_from_pickle
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer.promise import MyFuture
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def run_relevance_classifier(document_scorer: DocumentScorer, ca_query_list: List[CAQuery]) \
        -> Iterator[Tuple[CAQuery, List[Tuple[str, MyFuture[DocumentScorerOutput]]]]]:
    docs: List[Tuple[str, TokenizedText]] = load_from_pickle("ca_run3_document_processed")
    docs_d = dict(docs)
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.txt")
    ranked_list_groups = load_ranked_list_grouped(rlg_path)
    tokenizer = get_tokenizer()
    output: List[Tuple[CAQuery, List[Tuple[str, MyFuture]]]] = []
    for ca_query in ca_query_list:
        qid = ca_query.qid
        query_text = ca_query.ca_query
        ranked_list = ranked_list_groups[qid]
        doc_id_list = list(map(TrecRankedListEntry.get_doc_id, ranked_list))
        doc_id_list = doc_id_list[:1000]

        def do_score(doc_id) -> Tuple[str, MyFuture[DocumentScorerOutput]]:
            doc = docs_d[doc_id]
            q_tokens: List[str] = tokenizer.tokenize(query_text)
            sdp: MyFuture[DocumentScorerOutput] = document_scorer.score_relevance(q_tokens, doc)
            return doc_id, sdp

        res: List[Tuple[str, MyFuture]] = list(map(do_score, doc_id_list))
        output.append((ca_query, res))
    document_scorer.pk.do_duty()
    return output


def main():
    model_path = sys.argv[1]
    tsv_path = sys.argv[2]
    ca_query_list: List[CAQuery] = load_ca_query_from_tsv(tsv_path)
    document_scorer = DocumentScorer(Predictor(model_path, 2))
    output: Iterator[Tuple[CAQuery, List[Tuple[str, MyFuture[DocumentScorerOutput]]]]]\
        = run_relevance_classifier(document_scorer, ca_query_list)
    n_q_skip = 0
    skip_idx = 0
    save_data: List[Tuple[CAQuery, _]] = []
    for ca_query, docs_and_scores_future in output:
        print(ca_query)
        if skip_idx < n_q_skip:
            print("skip")
            skip_idx += 1
            continue
        docs_and_scores = [(doc_id, doc_scores_future.get())for doc_id, doc_scores_future in docs_and_scores_future]
        save_data.append((ca_query, docs_and_scores))

        # for doc_id, doc_scores_future in docs_and_scores_future:
        #     doc_scores: DocumentScorerOutput = doc_scores_future.get()
        #     high_scores = list(filter(lambda x: x > 0.5, doc_scores.scores))
        #     print(f"{doc_id}: {len(high_scores)} / {len(doc_scores)}")
    save_path = tsv_path + ".result"
    pickle.dump(save_data, open(save_path, "wb"))


if __name__ == "__main__":
    main()

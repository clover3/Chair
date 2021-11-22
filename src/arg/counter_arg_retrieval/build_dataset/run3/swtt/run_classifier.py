import os
import pickle
import sys
from typing import List, Tuple, Iterator

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from bert_api.msmarco_tokenization import EncoderUnit
from bert_api.predictor import Predictor, FloatPredictor, PredictorWrap
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer import DocumentScorerSWTT
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from bert_api.swtt.window_enum_policy import WindowEnumPolicy
from cache import load_from_pickle
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right
from trainer.promise import MyFuture
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry

ScoredDocument = Tuple[str, MyFuture[SWTTScorerOutput]]


def run_classifier(docs, document_scorer: DocumentScorerSWTT,
                   enum_policy: WindowEnumPolicy,
                   ca_query_list: List[CAQuery]) \
        -> Iterator[Tuple[CAQuery, List[ScoredDocument]]]:
    n_docs = len(docs)
    duplicate_indices = SegmentwiseTokenizedText.get_duplicate(right(docs))
    print("duplicate_indices {} ".format(len(duplicate_indices)))
    duplicate_doc_ids = [docs[idx][0] for idx in duplicate_indices]
    docs = [e for idx, e in enumerate(docs) if idx not in duplicate_indices]
    print("{} docs after filtering (from {})".format(len(docs), n_docs))

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
        doc_id_list = [doc_id for doc_id in doc_id_list if doc_id not in duplicate_doc_ids]
        doc_id_list = doc_id_list[:1000]

        def do_score(doc_id) -> Tuple[str, MyFuture[SWTTScorerOutput]]:
            doc = docs_d[doc_id]
            q_tokens: List[str] = tokenizer.tokenize(query_text)
            sdp: MyFuture[SWTTScorerOutput] = document_scorer.score(q_tokens, doc, enum_policy.window_enum)
            return doc_id, sdp

        res: List[Tuple[str, MyFuture]] = list(map(do_score, doc_id_list))
        output.append((ca_query, res))
    document_scorer.pk.do_duty()
    return output


def run_batch_predictions(ca_query_list, docs, predictor: FloatPredictor, enum_policy, save_path):
    document_scorer = DocumentScorerSWTT(predictor, EncoderUnit, 512)
    output: Iterator[Tuple[CAQuery, List[ScoredDocument]]] \
        = run_classifier(docs, document_scorer, enum_policy, ca_query_list)
    n_q_skip = 0
    skip_idx = 0
    save_data: List[Tuple[CAQuery, List[ScoredDocument]]] = []
    for ca_query, docs_and_scores_future in output:
        print(ca_query)
        if skip_idx < n_q_skip:
            print("skip")
            skip_idx += 1
            continue
        docs_and_scores = [(doc_id, doc_scores_future.get()) for doc_id, doc_scores_future in docs_and_scores_future]
        save_data.append((ca_query, docs_and_scores))

    pickle.dump(save_data, open(save_path, "wb"))


def main():
    model_path = sys.argv[1]
    tsv_path = sys.argv[2]
    ca_query_list: List[CAQuery] = load_ca_query_from_tsv(tsv_path)
    save_path = tsv_path + ".result"
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    enum_policy = WindowEnumPolicy(0, 10)
    predictor: FloatPredictor = PredictorWrap(Predictor(model_path, 2), lambda x: x[1])
    run_batch_predictions(ca_query_list, docs, predictor, enum_policy, save_path)


if __name__ == "__main__":
    main()

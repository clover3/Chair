from typing import List, Tuple, Dict, Iterable

from arg.qck.trec_helper import score_d_to_trec_style_predictions
from bert_api.client_lib_msmarco import BERTClientMSMarco, get_localhost_client
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from list_lib import lmap, right, get_max_idx
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, promise_to_items
from trec.types import TrecRankedListEntry


# Check if any segment is relevant to the query

def rerank_with_msmarco(queries: List[Tuple[str, str]],
                        docs_d: Dict[str, str],
                        rlg: Dict[str, List[TrecRankedListEntry]],
                        ) -> Tuple[Iterable[TrecRankedListEntry], Dict[Tuple[str, str], str]]:
    max_seq_length = 512
    print("ranked list contains {} queries".format(len(rlg)))
    client = get_msmarco_client()
    tokenizer = get_tokenizer()
    pk = PromiseKeeper(client.send_payload)
    ##
    q_and_docs = []
    for qid, query_text in queries:
        q_tokens = tokenizer.tokenize(query_text)
        available_doc_len = max_seq_length - len(q_tokens) - 3
        print(qid)
        try:
            for e in rlg[qid]:
                doc_text = docs_d[e.doc_id]
                doc_tokens = tokenizer.tokenize(doc_text)
                segment_list: List[List[str]] = split_window(doc_tokens, available_doc_len)

                def seg_to_promise(segment):
                    e = client.encoder.encode_token_pairs(q_tokens, segment)
                    promise = MyPromise(e, pk)
                    return promise

                promise_list = lmap(seg_to_promise, segment_list)
                e = qid, e.doc_id, promise_list, segment_list
                q_and_docs.append(e)
        except KeyError as e:
            print(e)
    print("Sending request")
    pk.do_duty(True)
    print("Request done")
    score_d: Dict[Tuple[str, str], float] = {}
    rel_hint = {}
    for qid, doc_id, promise_list, segment_list in q_and_docs:
        prob_pairs: List[Tuple[float, float]] = promise_to_items(promise_list)
        probs = right(prob_pairs)
        max_score = max(probs) if probs else 0
        if max_score:
            max_idx = get_max_idx(probs)
            tokens = segment_list[max_idx]
            rel_text = pretty_tokens(tokens, True)
        else:
            rel_text = ""
        rel_hint[(qid, doc_id)] = rel_text
        score_d[qid, doc_id] = max_score
    tr_entries: Iterable[TrecRankedListEntry] = score_d_to_trec_style_predictions(score_d, "MSMARCO")
    return list(tr_entries), rel_hint


def get_msmarco_client() -> BERTClientMSMarco:
    return get_localhost_client()

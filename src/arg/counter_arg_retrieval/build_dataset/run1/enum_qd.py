from typing import List, Dict, Tuple

from bert_api.msmarco_rerank import get_msmarco_client
from data_generator.common import get_tokenizer
from list_lib import lmap, right
from log_lib import log_var_len
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, promise_to_items
from trec.types import TrecRankedListEntry


def show_rerank_with_msmarco(queries: List[Tuple[str, str]],
                        docs_d: Dict[str, str],
                        rlg: Dict[str, List[TrecRankedListEntry]],
                        ):
    max_seq_length = 512
    print("ranked list contains {} queries".format(len(rlg)))
    client = get_msmarco_client()
    tokenizer = get_tokenizer()
    ##

    for qid, query_text in queries:
        q_tokens = tokenizer.tokenize(query_text)
        available_doc_len = max_seq_length - len(q_tokens) - 3
        try:
            pk = PromiseKeeper(client.send_payload)
            docs = []
            for e in rlg[qid]:
                doc_text = docs_d[e.doc_id]
                doc_tokens = tokenizer.tokenize(doc_text)
                segment_list: List[List[str]] = split_window(doc_tokens, available_doc_len)

                def seg_to_promise(segment):
                    e = client.encoder.encode_token_pairs(q_tokens, segment)
                    promise = MyPromise(e, pk)
                    return promise

                promise_list = lmap(seg_to_promise, segment_list)
                e = e.doc_id, promise_list
                docs.append(e)
            print("qid: ", qid)
            pk.do_duty(True)
            log_var_len(rlg[qid])

            for doc in docs:
                doc_id, promise_list = doc
                prob_pairs: List[Tuple[float, float]] = promise_to_items(promise_list)
                probs = right(prob_pairs)
                max_score = max(probs) if probs else 0
                print(doc_id, max_score)

        except KeyError as e:
            print(e)

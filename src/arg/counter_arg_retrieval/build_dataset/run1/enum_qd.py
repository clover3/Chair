import os
from typing import List, Dict, Tuple, Iterable

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import ca_building_q_res_path, \
    load_step2_claims_as_ca_topic, load_run1_doc_indexed
from bert_api.msmarco_rerank import rerank_with_msmarco, get_msmarco_client
from cpath import output_path
from data_generator.common import get_tokenizer
from list_lib import lfilter, lmap, right
from log_lib import log_var_len
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, promise_to_items
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def show_rerank_with_msmarco(queries: List[Tuple[str, str]],
                        docs_d: Dict[str, str],
                        rlg: Dict[str, List[TrecRankedListEntry]],
                        ) -> Iterable[TrecRankedListEntry]:
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


def main():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    rlg = load_ranked_list_grouped(ca_building_q_res_path)
    docs_d = load_run1_doc_indexed()
    topics = lfilter(lambda topic: topic.ca_cid in rlg, topics)

    def topic_to_query(topic: CaTopic):
        query = topic.claim_text + " " + topic.p_text
        qid = str(topic.ca_cid)
        return qid, query

    queries: List[Tuple[str, str]] = lmap(topic_to_query, topics)

    # GOAL : Select documents that are most relevant to the topic
    print("After filtering")
    log_var_len(topics)
    save_path = os.path.join(output_path, "ca_building", "run1", "msmarco_ranked_list.txt")
    tr_entries = rerank_with_msmarco(queries, docs_d, rlg)
    write_trec_ranked_list_entry(tr_entries, save_path)


if __name__ == "__main__":
    main()

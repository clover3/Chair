import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import ca_building_q_res_path, \
    load_step2_claims_as_ca_topic, load_run1_doc_indexed
from bert_api.msmarco_rerank import rerank_with_msmarco
from cpath import output_path
from list_lib import lfilter, lmap
from log_lib import log_var_len
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


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
    tr_entries, rel_hint = rerank_with_msmarco(queries, docs_d, rlg)
    write_trec_ranked_list_entry(tr_entries, save_path)


if __name__ == "__main__":
    main()

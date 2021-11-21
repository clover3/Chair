import os
from collections import defaultdict
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import ca_building_q_res_path, \
    load_step2_claims_as_ca_topic, load_run1_doc_indexed
from arg.counter_arg_retrieval.build_dataset.run2.load_data import load_my_run2_topics
from arg.counter_arg_retrieval.build_dataset.run2.write_q_res_html import load_cached_title
from bert_api.msmarco_rerank import rerank_with_msmarco
from cache import save_to_pickle
from cpath import output_path
from list_lib import lmap
from log_lib import log_var_len
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def regroup_ranked_list(rlg_run1, run1_qid_index, run2_queries):
    rlg_run2 = {}
    for run2_topic in run2_queries:
        run1_qids = run1_qid_index[run2_topic.claim]
        if not run1_qids:
            print("WARNING this topic is not found from run1")
            print(run2_topic.claim)
        all_docs = set()
        for qid in run1_qids:
            for entry in rlg_run1[qid]:
                all_docs.add(entry.doc_id)
        run2_entries = []
        for rank, doc_id in enumerate(all_docs):
            e = TrecRankedListEntry(run2_topic.qid, doc_id, rank, 0, "ZeroScore")
            run2_entries.append(e)
        rlg_run2[run2_topic.qid] = run2_entries
    return rlg_run2


def filter_rlg_by_title(rlg):
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    title_d_all = load_cached_title(topics)

    def filter_entries(ranked_list):
        seen = set()
        output = []
        for e in ranked_list:
            title = title_d_all[e.doc_id]
            if title not in seen:
                seen.add(title)
                output.append(e)
        return output

    return {k: filter_entries(v) for k, v in rlg.items()}


def rerank_and_save(save_name, topic_to_query):
    run2_topics = load_my_run2_topics()
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    QID = str
    Claim = str
    run1_qid_index: Dict[Claim, List[QID]] = defaultdict(list)
    for topic in topics:
        run1_qid_index[topic.claim_text].append(topic.ca_cid)
    rlg_run1 = load_ranked_list_grouped(ca_building_q_res_path)
    rlg_run2 = regroup_ranked_list(rlg_run1, run1_qid_index, run2_topics)
    rlg = filter_rlg_by_title(rlg_run2)

    docs_d = load_run1_doc_indexed()
    queries: List[Tuple[str, str]] = lmap(topic_to_query, run2_topics)
    # GOAL : Select documents that are most relevant to the topic
    print("After filtering")
    log_var_len(run2_topics)
    # save_name_format.format(qid)
    save_path = os.path.join(output_path, "ca_building", "run2", save_name)
    tr_entries, rel_hint = rerank_with_msmarco(queries, docs_d, rlg)
    write_trec_ranked_list_entry(tr_entries, save_path)
    save_to_pickle(rel_hint, save_name + ".rel_hint")


def concat():
    def topic_to_query(topic: CAQuery):
        query = topic.claim + " " + topic.ca_query
        return topic.qid, query
    save_name = "rerank_concat_{}.txt"

    rerank_and_save(save_name, topic_to_query)


def ca_only():
    def topic_to_query(topic: CAQuery):
        query = topic.ca_query
        return topic.qid, query
    save_name = "rerank_ca.txt"

    rerank_and_save(save_name, topic_to_query)


def pers_query():
    def topic_to_query(topic: CAQuery):
        query = topic.perspective
        return topic.qid, query
    save_name = "rerank_pers.txt"

    rerank_and_save(save_name, topic_to_query)


if __name__ == "__main__":
    pers_query()

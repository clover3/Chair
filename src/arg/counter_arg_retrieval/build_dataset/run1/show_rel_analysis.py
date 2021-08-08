import json
import os
from typing import List

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic, CaTopicv2, get_ca2_converter
from arg.counter_arg_retrieval.build_dataset.resources import ca_building_q_res_path, \
    load_step2_claims_as_ca_topic
from arg.counter_arg_retrieval.scorer.relevance_analysis import AnalyzedDocument
from cache import load_pickle_from
from cpath import output_path
from list_lib import lfilter, lmap
from trec.trec_parse import load_ranked_list_grouped


def topic_and_ad_list_to_table(topic: CaTopicv2, ad_list: List[AnalyzedDocument], max_segs):
    # for each document
    rows = []
    for document in ad_list:
        num_p = len(document.p_scores)
        assert len(topic.other_ps) + 1 == num_p
        num_segs = min(len(document.c_scores.scores), max_segs)
        rows.append(["Document"])
        # for each segment
        row = ["C"]
        for seg_idx in range(num_segs):
            # list all relevant claim/perpsective
            c_score: float = document.c_scores.scores[seg_idx]
            if c_score > 0.5:
                row.append(seg_idx)
        rows.append(row)

        for p_idx in range(num_p):
            row = []
            for seg_idx, score in enumerate(document.p_scores[p_idx].scores):
                if score > 0.5:
                    row.append(seg_idx)
            if row:
                row = ["P{}".format(p_idx)] + row
                rows.append(row)
        rows.append(["Num perspective: {}".format(num_p)])
    return rows


def print_relevant_parts(topic: CaTopicv2, ad_list: List[AnalyzedDocument], max_segs):
    def get_st_ed(window_start_loc: List[int], seg_idx):
        st = window_start_loc[seg_idx]
        try:
            ed = window_start_loc[seg_idx+1]
        except IndexError:
            ed = st + 10000
        return st, ed

    doc_j_out = []
    for document in ad_list:
        num_p = len(document.p_scores)
        assert len(topic.other_ps) + 1 == num_p
        num_segs = min(len(document.c_scores.scores), max_segs)
        j_document = {}
        # for each segment

        c_rel_list = []
        for seg_idx in range(num_segs):
            # list all relevant claim/perpsective
            c_score: float = document.c_scores.scores[seg_idx]
            if c_score > 0.5:
                st, ed = get_st_ed(document.c_scores.window_start_loc, seg_idx)
                text = " ".join(document.dot_tokens[st:ed])
                c_rel_list.append(text)

        j_document['claim'] = topic.claim_text
        j_document['claim_rel'] = c_rel_list

        p_list = [topic.target_p] + topic.other_ps
        for p_idx in range(num_p):
            p_rel_list = []
            for seg_idx, score in enumerate(document.p_scores[p_idx].scores):
                if score > 0.5:
                    st, ed = get_st_ed(document.p_scores[p_idx].window_start_loc, seg_idx)
                    text = " ".join(document.dot_tokens[st:ed])
                    p_rel_list.append((st, ed, text))
            if p_rel_list:
                pid, p_text = p_list[p_idx][0]
                j_document["P{}".format(p_idx)] = p_text
                j_document["P{}_rel".format(p_idx)] = p_rel_list

        doc_j_out.append(j_document)
    per_topic_j = {
        'ca_cid': topic.ca_cid,
        'topic': topic.claim_text,
        'docs': doc_j_out
    }
    return per_topic_j


def load_rel(topic) -> List[AnalyzedDocument]:
    dir_path = os.path.join(output_path, "ca_building", "run1", "relevance_analysis")
    return load_pickle_from(os.path.join(dir_path, topic.ca_cid))


def main():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    rlg = load_ranked_list_grouped(ca_building_q_res_path)
    topics: List[CaTopic] = lfilter(lambda topic: topic.ca_cid in rlg, topics)
    topics_v2: List[CaTopicv2] = lmap(get_ca2_converter(), topics)
    max_segs = 20
    j_topics = []
    for topic_idx, topic in enumerate(topics_v2):
        try:
            ad_list = load_rel(topic)
            print("Topic {}".format(topic_idx))
            per_topic_j = print_relevant_parts(topic, ad_list, max_segs)
            j_topics.append(per_topic_j)
        except FileNotFoundError:
            break
    output_d = {
        'topics': j_topics
    }
    j_out_path = os.path.join(output_path, "ca_building", "run1", "relevance_analysis", "summary.json")

    json.dump(output_d, open(j_out_path, "w"), indent=True)


if __name__ == "__main__":
    main()

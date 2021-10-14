import os
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic
from arg.counter_arg_retrieval.build_dataset.run1.write_q_res_html import write_htmls
from arg.counter_arg_retrieval.build_dataset.run2.load_data import load_run2_topics, CAQuery
from cache import load_json_cache, load_from_pickle
from cpath import output_path
from misc_lib import exist_or_mkdir
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    run2_topics: List[CAQuery] = load_run2_topics()
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    query_gen_method = "ca"
    qid_to_topic = {t.qid: t for t in run2_topics}
    urls_d = load_json_cache("ca_urls_d")
    save_name = "rerank_{}.txt".format(query_gen_method)
    ranking1 = os.path.join(output_path,
                            "ca_building",
                            "run2",
                            save_name)
    rel_hint = load_from_pickle(save_name + '.rel_hint')
    # rel_hint = load_from_pickle('rel_hint')
    html_doc_root = os.path.join(output_path, "ca_building", "run1", "html")
    rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking1)
    out_root = os.path.join(output_path,
                            "ca_building",
                            "run2",
                            "doc_list_html_{}".format(query_gen_method))
    exist_or_mkdir(out_root)
    title_d_all = load_cached_title(topics)

    def qid_to_headlines(qid):
        topic = qid_to_topic[qid]
        return [
            "Claim: {}".format(topic.claim),
            "Perspective: {}".format(topic.perspective),
            "Counter-arg: {}".format(topic.ca_query)
        ]

    write_htmls(rlg,
                out_root,
                qid_to_headlines,
                urls_d,
                title_d_all,
                rel_hint,
                html_doc_root)


def load_cached_title(topics):
    title_d_all = {}
    for topic in topics:
        q_id = topic.ca_cid
        title_cache = "title_d_ca_{}".format(q_id)
        title_d = load_json_cache(title_cache)
        if title_d is not None:
            title_d_all.update(title_d)
    return title_d_all


if __name__ == "__main__":
    main()
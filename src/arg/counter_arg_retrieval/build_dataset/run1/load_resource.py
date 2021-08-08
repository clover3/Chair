from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic, CaTopicv2, get_ca2_converter
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic, ca_building_q_res_path, \
    load_run1_doc_indexed
from list_lib import lfilter, lmap
from trec.trec_parse import load_ranked_list_grouped


def get_run1_resource():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    rlg = load_ranked_list_grouped(ca_building_q_res_path)
    docs_d: Dict[str, str] = load_run1_doc_indexed()
    topics: List[CaTopic] = lfilter(lambda topic: topic.ca_cid in rlg, topics)
    topics_v2: List[CaTopicv2] = lmap(get_ca2_converter(), topics)
    return rlg, topics_v2, docs_d
import json
import os
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from cache import load_pickle_from
from cpath import at_output_dir, output_path
from list_lib import lmap


def load_step1_claims():
    j_obj = json.load(open(at_output_dir("ca_building", "claims.step1.txt"), "r"))
    return j_obj


def load_step2_claims():
    j_obj = json.load(open(os.path.join(output_path, "ca_building", "run1", "claims.run1.txt"), "r"))
    return j_obj


def load_step2_claims_as_ca_topic() -> List[CaTopic]:
    j_obj = json.load(open(os.path.join(output_path, "ca_building", "run1", "claims.run1.txt"), "r"))
    return lmap(CaTopic.from_j_entry, j_obj)


ca_building_q_res_path = os.path.join(output_path, "ca_building", "q_res", "q_res_all")


def load_run1_cleaned_docs():
    return load_pickle_from(os.path.join(output_path, "ca_building", "run1", "docs_cleaned.pickle"))


def load_run1_doc_indexed() -> Dict[str, str]:
    docs = load_run1_cleaned_docs()
    out_d = {}
    for d in docs:
        out_d[d['doc_id']] = d['core_text']
    return out_d

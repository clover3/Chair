import random
from typing import List, Dict, NamedTuple

from cache import dump_to_json
from dataset_specific.msmarco.common import QueryID
from dataset_specific.msmarco.misc_tool import get_qid_to_job_id
from list_lib import flatten
from misc_lib import group_by, tprint
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListPredict


class QDPair(NamedTuple):
    qid: str
    doc_id: str
    q_tokens: List[int]
    d_tokens: List[int]
    score: float


def main():
    # read predictions
    # For each of Relevant/Non-relevant
    n_qids = 1000
    split = "dev"
    resource_source = ProcessedResourceTitleBodyTokensListPredict(split)
    qid_to_job_id = get_qid_to_job_id(resource_source.query_group)
    all_qids = list(flatten(resource_source.query_group))
    selected_qids: List[QueryID] = random.sample(all_qids, n_qids)
    tprint("Selected QIDs")
    selected_qids_grouped: Dict[int, List[QueryID]] = group_by(selected_qids, lambda k: qid_to_job_id[k])
    dump_to_json(selected_qids_grouped, "MMD_selected_qids")


if __name__ == "__main__":
    main()

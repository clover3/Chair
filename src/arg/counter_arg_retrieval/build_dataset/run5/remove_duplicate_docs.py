# Remove duplicate documents
import os
from typing import List, Dict, Tuple

# Input: Ranked List, SWTT
# Output: Ranked List
from arg.counter_arg_retrieval.build_dataset.data_prep.duplicate_removal import remove_duplicates
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_swtt_jsonl_per_query_as_d
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import flatten
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


class Worker(WorkerInterface):
    def __init__(self, rlg: Dict[str, List[TrecRankedListEntry]], job_path):
        self.rlg: Dict[str, List[TrecRankedListEntry]] = rlg
        self.job_path = job_path
        qids = list(self.rlg.keys())
        qids.sort()
        self.qids = qids

    def work(self, job_no):
        query_id = self.qids[job_no]
        docs: Dict[str, SegmentwiseTokenizedText] = load_swtt_jsonl_per_query_as_d(query_id)
        docs_list: List[Tuple[str, SegmentwiseTokenizedText]] = [(k, v) for k, v in docs.items()]
        rlg_part = {k: self.rlg[k] for k in [query_id]}
        docs, duplicate_doc_ids, new_rlg = remove_duplicates(rlg_part, docs_list)
        print("Duplicate rate: {0:.2f}".format(len(duplicate_doc_ids) / len(docs)))
        rl_itr = flatten(new_rlg.values())
        rlg_path = os.path.join(self.job_path, query_id)
        write_trec_ranked_list_entry(rl_itr, rlg_path)


def main():
    rlg_path = os.path.join(output_path, "ca_building", "run5", "q_res.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    num_jobs = len(rlg)
    runner = JobRunnerS(job_man_dir, num_jobs, "ca_building_run5_filter_duplicate", lambda p: Worker(rlg, p))
    runner.start()


if __name__ == "__main__":
    main()
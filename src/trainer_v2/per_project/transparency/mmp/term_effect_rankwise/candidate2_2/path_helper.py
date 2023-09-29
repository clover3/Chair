from cpath import output_path
from misc_lib import path_join


def get_candidate2_2_term_pair_candidate_building_path(job_no):
    save_path = path_join(
        output_path, "msmarco", "passage",
        "candidate2_2_building", f"{job_no}.jsonl")
    return save_path
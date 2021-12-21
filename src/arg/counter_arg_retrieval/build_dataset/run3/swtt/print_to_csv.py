import os
from typing import List, Tuple
from typing import NamedTuple

from arg.counter_arg_retrieval.build_dataset.annotation_prep import read_save
from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cache import load_pickle_from
from cpath import output_path
from misc_lib import get_dir_files
from trec.trec_parse import load_ranked_list_grouped


def tuple_compare(tuple1, tuple2):
    num_dim = 5
    for j in range(num_dim):
        v1 = tuple1[j]
        v2 = tuple2[j]
        if v1 != v2:
            if v1 < v2:
                return -1
            else:
                return +1
    return 0


class JudgeEntry(NamedTuple):
    p_text: str
    c_text: str
    doc_id: str
    passage_idx: int
    passage_html: str


class PassageID(NamedTuple):
    doc_id: str
    passage_idx: int


def get_candidate_passages(docs_and_scores, duplicate_doc_ids) -> List[Tuple[PassageID, float]]:
    judge_candidates = []
    for doc_id, scores in docs_and_scores:
        if doc_id in duplicate_doc_ids:
            continue
        for passage_idx, score in enumerate(scores.scores):
            e = PassageID(doc_id, passage_idx), score
            judge_candidates.append(e)
    return judge_candidates



def read_save_default(run_name):
    csv_save_path = os.path.join(output_path, "ca_building", "run3", "csv", "{}.csv".format(run_name))
    prediction_entries = load_entries_from_jobs_dir(run_name)
    sliced_ranked_list_path = os.path.join(output_path, "ca_building",
                                           "run3", "passage_ranked_list_sliced",
                                           "{}.txt".format(run_name))
    ranked_list_d = load_ranked_list_grouped(sliced_ranked_list_path)
    read_save(prediction_entries, ranked_list_d, csv_save_path)


def load_entries_from_jobs_dir(run_name):
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = []
    save_dir = os.path.join(output_path, "ca_building", "jobs", run_name)
    for file_path in get_dir_files(save_dir):
        prediction_entries.extend(load_pickle_from(file_path))
    return prediction_entries


def load_json_entries_from_jobs_dir(run_name, parse_fn):
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = []
    save_dir = os.path.join(output_path, "ca_building", "jobs", run_name)
    for file_path in get_dir_files(save_dir):
        if file_path.endswith(".json"):
            prediction_entries.extend(parse_fn(file_path))
    return prediction_entries


def main():
    read_save_default("PQ_1")


if __name__ == "__main__":
    main()

